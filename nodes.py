"""ComfyUI nodes for AReA (Adaptive Receptive-Field Attention).

- AReAModelPatch:       MODEL -> MODEL, attaches AReA attention via official
                        ModelPatcher extension points (no ComfyUI source changes).
                        Auto-detects the backbone: FLUX.1, Qwen-Image or Ideogram 4
                        checkpoints loaded through ComfyUI's native loaders.
- AReABootstrapSampler: SAMPLER, AReA-Bootstrap two-stage sampling (native-res
                        generation -> latent upsample -> re-noise -> Phase II),
                        implemented on the native ComfyUI sampling loop (FLUX.1 only).
"""

import torch
import torch.nn.functional as F
from tqdm.auto import trange

import comfy.patcher_extension
import comfy.samplers

from . import comfy_backend
from .comfy_backend import AReAConfig, DEFAULT_PHASE1_STEPS


def _euler_loop(model_k, x, sigmas, extra_args, callback, disable, offset, total_steps):
    """Native ComfyUI flow-matching Euler loop (matches comfy sample_euler, s_churn=0)."""
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i] * s_in
        denoised = model_k(x, sigma, extra_args.get("denoise_mask"),
                           model_options=extra_args.get("model_options", {}),
                           seed=extra_args.get("seed", None))
        if callback is not None:
            callback({"x": x, "i": offset + i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})
        d = (x - denoised) / sigmas[i]
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


def _sample_euler_plain(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
    """Plain euler sampler function (KSAMPLER contract: model_k is a prepared KSamplerX0Inpaint)."""
    extra_args = {} if extra_args is None else extra_args
    if len(sigmas) <= 1:
        return noise
    total = len(sigmas) - 1

    def cb(d):
        if callback is not None:
            callback(d["i"], d["denoised"], d["x"], total)

    return _euler_loop(model_k, noise, sigmas, extra_args, cb, disable, 0, total)


class AReABootstrapKSampler(comfy.samplers.KSAMPLER):
    """AReA-Bootstrap: stage 1 generates at the native resolution with plain attention, the
    latent is bicubically upsampled to the target size, re-noised at the level where Phase II
    begins, and denoised with Phase II windowed attention (AReA paper, Appendix F)."""

    def __init__(self, bootstrap_side=1024, phase1_steps=DEFAULT_PHASE1_STEPS):
        self.bootstrap_side = int(bootstrap_side)
        self.phase1_steps = int(phase1_steps)
        super().__init__(_sample_euler_plain)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        extra_args = dict(extra_args)
        extra_args["denoise_mask"] = denoise_mask
        model_k = comfy.samplers.KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise
        ms = model_wrap.inner_model.model_sampling
        total_steps = len(sigmas) - 1
        if total_steps < 2:
            return noise

        def cb(d):
            if callback is not None:
                callback(d["i"], d["denoised"], d["x"], total_steps)

        max_denoise = self.max_denoise(model_wrap, sigmas)
        x = ms.noise_scaling(sigmas[0], noise, latent_image, max_denoise)

        bootstrap = (denoise_mask is None and latent_image is not None and max_denoise
                     and not bool(torch.count_nonzero(latent_image)))
        if not bootstrap:
            # img2img / inpaint: plain euler at the given resolution
            x = _euler_loop(model_k, x, sigmas, extra_args, cb, disable_pbar, 0, total_steps)
            return ms.inverse_noise_scaling(sigmas[-1], x)

        # ---- stage 1: native-resolution generation (grid <= native extent -> plain attention)
        if noise.dim() == 5:                      # video-style latents, e.g. Qwen-Image [B, C, T, H, W]
            b, c, fr, hl, wl = noise.shape
        else:
            b, c, hl, wl = noise.shape
            fr = None
        # latent pixels-per-unit of this model: Ideogram 4's latent IS the patch grid (/16),
        # the others (FLUX.1, Qwen-Image) are /8 VAE latents
        dm = getattr(model_wrap.inner_model, "diffusion_model", None)
        divisor = 16 if type(dm).__name__ == "Ideogram4Transformer2DModel" else 8
        side = max(2, int(round(self.bootstrap_side / divisor)))    # latent side in px / divisor
        scale = min(1.0, side / float(max(hl, wl)))
        low_h = max(2, int(round(hl * scale)) // 2 * 2)
        low_w = max(2, int(round(wl * scale)) // 2 * 2)
        seed = int(extra_args.get("seed", 0) or 0)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        low_shape = (b, c, low_h, low_w) if fr is None else (b, c, fr, low_h, low_w)
        x_low = torch.randn(low_shape, generator=gen, device="cpu", dtype=torch.float32)
        x_low = x_low.to(device=noise.device, dtype=noise.dtype)
        x_clean = _euler_loop(model_k, x_low, sigmas, extra_args, cb, disable_pbar, 0, total_steps)

        # ---- stage 2: latent upsample -> re-noise at the Phase II start level -> Phase II
        p1 = int(min(max(0, self.phase1_steps), total_steps - 1))
        x_up = x_clean.float()
        if fr is None:
            clean = F.interpolate(x_up, size=(hl, wl), mode="bicubic", align_corners=False)
        elif fr == 1:
            clean = F.interpolate(x_up.squeeze(2), size=(hl, wl), mode="bicubic",
                                  align_corners=False).unsqueeze(2)
        else:
            clean = F.interpolate(x_up, size=(fr, hl, wl), mode="trilinear", align_corners=False)
        clean = clean.to(device=noise.device, dtype=noise.dtype)
        gen.manual_seed(seed)
        noise2 = torch.randn(clean.shape, generator=gen, device="cpu", dtype=torch.float32)
        noise2 = noise2.to(device=noise.device, dtype=noise.dtype)
        sig2 = sigmas[p1:]
        x2 = ms.noise_scaling(sig2[0], noise2, clean, False)
        x2 = _euler_loop(model_k, x2, sig2, extra_args, cb, disable_pbar, p1, total_steps)
        return ms.inverse_noise_scaling(sigmas[-1], x2)


class AReAModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "pooling": (comfy_backend.area_attention.POOLING, {"default": "subsample",
                            "tooltip": "Phase I: which token represents each f x f block."}),
                "phase1_steps": ("INT", {"default": DEFAULT_PHASE1_STEPS, "min": 0, "max": 64, "step": 1,
                                         "tooltip": "Phase I (cross-scale coarse attention) steps. Recommended: "
                                                    "FLUX.1 8 of 28; Qwen-Image ~20 of 40 (4096px+); "
                                                    "Ideogram 4 ~20 of 48. Set 0 for img2img super-resolution "
                                                    "(structure already exists, Phase II only)."}),
                "halo": ("INT", {"default": 8, "min": 0, "max": 32, "step": 1,
                                 "tooltip": "Phase II: largest Halo width, in patches."}),
                "bridge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Phase II: each window gets (ratio * w)^2 global-KV bridge tokens."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Draws window boundary jitter, Halo widths and random pooling."}),
                "native_patches": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1,
                                           "tooltip": "Native training extent in patches per side; 0 = per-model default "
                                                      "(FLUX.1: 64 = 1024 px, Qwen-Image: 83 = 1328 px, "
                                                      "Ideogram 4: 128 = 2048 px). Canvases at or below this keep "
                                                      "plain attention."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "AReA"
    DESCRIPTION = ("AReA: training-free adaptive receptive-field attention for high-resolution "
                   "FLUX.1 / Qwen-Image / Ideogram 4. Phase I pools K/V cross-scale with custom "
                   "block-coarse RoPE; Phase II runs adaptive windowed attention with Halo rings "
                   "and a global-KV bridge. Hooks only into official ModelPatcher extension points; "
                   "ComfyUI source is never modified.")

    def patch(self, model, pooling, phase1_steps, halo, bridge_ratio, seed, native_patches):
        dm = getattr(model.model, "diffusion_model", None)
        backend = comfy_backend.detect_backbone(dm) if dm is not None else None
        if backend is None:
            raise ValueError(
                "AReA requires a ComfyUI-native FLUX.1 (dev/schnell/krea), Qwen-Image or Ideogram 4 "
                f"checkpoint. Got diffusion model of type {type(dm).__name__ if dm is not None else None}.")
        m = model.clone()
        config = AReAConfig(pooling=pooling, phase1_steps=phase1_steps, halo=halo,
                            bridge_ratio=bridge_ratio, seed=seed, native_patches=native_patches)
        state = comfy_backend.AReAState(config, backend)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "AReA_step",
                               comfy_backend.make_wrapper(state))
        if backend == "flux":
            m.set_model_post_input_patch(comfy_backend.make_flux_post_input(state))
            m.set_model_attn1_patch(comfy_backend.make_flux_attn1_patch(state))
            m.set_model_attn1_output_patch(comfy_backend.make_flux_attn1_output_patch(state))
        elif backend == "qwen_image":
            m.set_model_post_input_patch(comfy_backend.make_qwen_post_input(state))
            m.set_model_attn1_patch(comfy_backend.make_qwen_attn1_patch(state))
            m.set_model_optimized_attention(comfy_backend.make_qwen_attention_override(state))
        else:
            m.set_model_optimized_attention(comfy_backend.make_ideogram_attention_override(state))
        return (m,)


class AReABootstrapSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bootstrap_side": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8,
                                           "tooltip": "Side (px) of the stage-1 native-resolution generation. "
                                                      "Set it to the model's native extent: FLUX.1 1024, "
                                                      "Qwen-Image 1328, Ideogram 4 2048."}),
                "phase1_steps": ("INT", {"default": DEFAULT_PHASE1_STEPS, "min": 1, "max": 64, "step": 1,
                                         "tooltip": "Where Phase II begins in the full step count. "
                                                    "Keep equal to phase1_steps of the AReA patch node."}),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "AReA"
    DESCRIPTION = ("AReA-Bootstrap sampler (FLUX.1 / Qwen-Image): stage 1 at native resolution, "
                   "latent upsampled and re-noised, stage 2 windowed fine attention at the target "
                   "resolution. Use with a KSampler / SamplerCustom whose model went through the "
                   "AReA patch node (denoise 1.0, txt2img). Keep phase1_steps equal to the patch "
                   "node's (FLUX.1: 8, Qwen-Image: ~20) and bootstrap_side at the model's native "
                   "resolution (FLUX.1: 1024, Qwen-Image: 1328).")

    def get_sampler(self, bootstrap_side, phase1_steps):
        return (AReABootstrapKSampler(bootstrap_side=bootstrap_side, phase1_steps=phase1_steps),)


NODE_CLASS_MAPPINGS = {
    "AReAModelPatch": AReAModelPatch,
    "AReABootstrapSampler": AReABootstrapSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AReAModelPatch": "AReA Patch (FLUX.1 / Qwen-Image / Ideogram 4 High-Res Attention)",
    "AReABootstrapSampler": "AReA Bootstrap Sampler",
}
