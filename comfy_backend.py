"""ComfyUI native-pipeline adapter for AReA (FLUX.1 / Qwen-Image / Ideogram 4).

Nothing here touches ComfyUI source; everything hooks through official ModelPatcher
extension points. One backbone per model family:

FLUX.1 (comfy.ldm.flux.model.Flux)
- ``wrappers[diffusion_model]``  -> locates the current denoising step from the model
  timestep and ``transformer_options["sample_sigmas"]``, keeps the Flux module.
- ``patches["post_input"]``      -> builds the per-step schedule state, the Phase I
  coarse RoPE tables and the Phase II fine layout (DWT detail map, adaptive windows,
  Halo, bridge).
- ``patches["attn1_patch"]``     -> Phase I: pools K/V to one token per f x f block and
  applies AReA's custom RoPE (queries quantised to block coords, keys on the coarse
  grid) with pe returned as None so the block's own attention runs on the pooled KV.
  Phase II: computes the windowed attention and stashes the result.
- ``patches["attn1_output_patch"]`` -> Phase II only: swaps the stashed windowed-
  attention result for the block's (deliberately trivialised) attention output.

Qwen-Image (comfy.ldm.qwen_image.model.QwenImageTransformer2DModel)
- ``wrappers[diffusion_model]`` and ``patches["post_input"]`` as above (the model's
  own attention supports ``attn1_patch`` before RoPE, like Flux).
- ``patches["attn1_patch"]``     -> Phase I: pools K/V + custom coarse RoPE, returns the
  already-rotated q/k and an identity RoPE table (the comfy Qwen-Image model applies
  RoPE unconditionally after the patch, so pe=None is not allowed).
- ``optimized_attention_override`` -> Phase II: comfy's Qwen-Image model has no
  ``attn1_output_patch`` hook, so the windowed attention replaces the whole attention
  call. The unrotated keys it needs for the bridge ranking are stashed per block by
  ``attn1_patch`` in the same block's forward.

Ideogram 4 (comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel)
- ``wrappers[diffusion_model]``  -> per-step state; the RoPE is fused into the model's
  RMSNorm+RoPE kernel before attention, so the override recovers unrotated keys with
  the inverse rotation instead of intercepting pre-RoPE tensors.
- ``optimized_attention_override`` -> both phases. Phase I: unrotate q/k with the true
  positions, pool the unrotated K/V, rotate q with block-quantised positions and the
  pooled keys with the coarse grid (text keeps its own positions). Phase II: windowed
  attention on the true-rotation q/k with unrotated keys for the bridge ranking.

RoPE math mirrors the comfy native implementations exactly (flux-style EmbedND for
FLUX/Qwen-Image; split-half MRoPE for Ideogram 4), matching the upstream diffusers
implementation of AReA. CFG branches share one batched forward on ComfyUI, which is
the same sharing the upstream ``memo`` machinery provides for true-CFG pipelines.
"""

import logging

import torch
import torch.nn.functional as F

from . import area
from .area import attention as area_attention
from .area import schedule as area_schedule
from .area import tiling as area_tiling

logger = logging.getLogger("comfy.custom_nodes.ComfyUI-AReA")

NATIVE_PATCHES = 64           # FLUX.1 native training extent: 1024 px / 16 px per patch
DEFAULT_PHASE1_STEPS = 8      # of 28
QWEN_NATIVE_PATCHES = 83      # Qwen-Image native extent: 1328 px / 16 px per patch
QWEN_WINDOW_FLOOR = 64        # Qwen-Image Phase II windows shrink to 1024 px
IDEOGRAM_NATIVE_PATCHES = 128  # Ideogram 4 native extent: 2048 px / 16 px per patch

BACKBONES = ("flux", "qwen_image", "ideogram4")
BACKBONE_LABELS = {"flux": "FLUX.1", "qwen_image": "Qwen-Image", "ideogram4": "Ideogram 4"}


def detect_backbone(dm):
    """Which AReA backbone fits this comfy diffusion model, or None."""
    name = type(dm).__name__
    if name == "Flux" and all(hasattr(dm, a) for a in ("double_blocks", "single_blocks", "pe_embedder", "params")):
        return "flux"
    if (name == "QwenImageTransformer2DModel" and hasattr(dm, "pe_embedder")
            and hasattr(dm, "transformer_blocks")):
        return "qwen_image"
    if name == "Ideogram4Transformer2DModel" and hasattr(dm, "layers") and hasattr(dm, "rope_theta"):
        return "ideogram4"
    return None


class AReAConfig:
    def __init__(self, pooling="subsample", phase1_steps=DEFAULT_PHASE1_STEPS,
                 halo=8, bridge_ratio=0.5, seed=0, native_patches=0):
        if pooling not in area_attention.POOLING:
            raise ValueError(f"pooling must be one of {area_attention.POOLING}, got {pooling!r}")
        self.pooling = pooling
        self.phase1_steps = int(phase1_steps)
        self.halo = int(halo)
        self.bridge_ratio = float(bridge_ratio)
        self.seed = int(seed)
        self.native_patches = int(native_patches)   # 0 = per-model default

    def native_for(self, backend):
        if self.native_patches > 0:
            return self.native_patches
        if backend == "flux":
            return NATIVE_PATCHES
        if backend == "qwen_image":
            return QWEN_NATIVE_PATCHES
        return IDEOGRAM_NATIVE_PATCHES


class AReAState:
    """Per-step denoising state shared by every patch of one patched model."""

    def __init__(self, config: AReAConfig, backend: str):
        self.config = config
        self.backend = backend
        self.native = config.native_for(backend)
        self.phase = None          # None: plain attention, "coarse": Phase I, "fine": Phase II
        self.step = None
        self.size = 0              # pooling factor f (Phase I) / window side w (Phase II)
        self.h = 0
        self.w = 0
        self.txt_len = 0           # text tokens of THIS forward (0 on an image-only pass)
        self.layout_txt_len = 0    # text tokens the schedule/layout is built with (shared per step)
        self.model = None          # the diffusion model module
        self.data = {}
        self._schedule = None
        self._schedule_key = None
        self._anchor = 0
        self._ropes = {}
        self.block_counter = 0     # Ideogram 4 has no block_index in transformer_options
        self._layout_cache = None
        self._layout_key = None
        self.last_txt_len = 0      # conditional pass's text length (Ideogram 4 uncond pass)
        self._is_last = False

    def reset_eval(self):
        """Called once per diffusion-model evaluation (one sampler step, both CFG branches)."""
        self.phase = None
        self.data = {}
        self.block_counter = 0

    # ---- step location & schedule -----------------------------------------

    def locate_step(self, timestep, topts):
        """(step index, sigmas) of the running call, from sample_sigmas (sigma units:
        every supported model runs a flow-matching model_sampling with multiplier 1.0)."""
        sigmas = topts.get("sample_sigmas", None)
        if sigmas is None:
            return None
        sigma_now = float(timestep.detach().reshape(-1)[0])
        sig = sigmas.detach().float().cpu()
        diffs = (sig - sigma_now).abs()
        idx = int(diffs.argmin())
        if float(diffs[idx]) > 1e-3 * max(1.0, sigma_now) + 1e-6:
            return None            # not on the registered schedule: plain attention
        return idx, sig

    def schedule(self, h, w, steps, txt_len):
        """The schedule for this canvas, or None at or below the native extent."""
        if self.backend == "ideogram4":
            extent = max(h, w)
            if extent <= self.native:
                return None
            # canvases up to 4096 px anchor on the half extent (1024 px), the native above
            anchor = self.native // 2 if extent <= 4 * (self.native // 2) else self.native
            w_min = None
        else:
            if max(h, w) <= self.native:
                return None
            anchor = self.native
            w_min = QWEN_WINDOW_FLOOR if self.backend == "qwen_image" else None
        phase1 = self.config.phase1_steps
        key = (h, w, steps, phase1, txt_len)
        if key != self._schedule_key:
            self._schedule = area_schedule.build_schedule(h, w, steps, phase1, anchor, w_min, txt_len=txt_len)
            self._schedule_key = key
            self._anchor = anchor
        return self._schedule

    def fine_layout(self, tokens, h, w, step, size, txt_len):
        """The Phase II layout of one step: DWT detail map -> adaptive windows -> Halo + bridge."""
        cfg = self.config
        w_min = min(self._anchor, max(h, w))
        energy = area_tiling.detail_map(tokens, h, w)
        windows = area_tiling.adaptive_tiling(
            energy, h, w, size, w_min, txt_len, rng=area_tiling.step_rng(cfg.seed, step, 0x5EA1))
        registers = (max(1, int(round(size * cfg.bridge_ratio))) ** 2
                     if cfg.bridge_ratio > 0 else 0)
        widths = area_tiling.halo_widths(windows, h, w, cfg.halo, area_tiling.step_rng(cfg.seed, step, 0x4A11))
        # never cost more than a full-attention step: cut bridge tokens, then the Halo
        registers, widths = area_attention.fit_extra_keys(windows, h, w, txt_len, registers, widths)
        return area_attention.build_fine_layout(
            windows, h, w, widths, cfg.halo, registers, cfg.seed, step, tokens.dtype, tokens.device)

    # ---- flux/qwen rotary embedding helpers --------------------------------

    def flux_coarse_ropes(self, h, w, f, txt_len, device):
        """Queries keep full resolution and share the coarse position of their block; pooled keys
        sit on the (h/f) x (w/f) grid, so both are measured in the same units."""
        key = ("coarse", h, w, f, txt_len, str(device))
        if self._ropes.get("coarse", (None,))[0] != key:
            pos = torch.arange(h * w, device=device)
            cpos = torch.arange((h // f) * (w // f), device=device)
            rope_q = self._flux_rope(txt_len, pos // w // f, pos % w // f, device)
            rope_kv = self._flux_rope(txt_len, cpos // (w // f), cpos % (w // f), device)
            self._ropes["coarse"] = (key, (rope_q, rope_kv))
        return self._ropes["coarse"][1]

    def _flux_rope(self, txt_len, rows, cols, device):
        ids = torch.zeros(1, txt_len + rows.numel(), 3, device=device, dtype=torch.float32)
        ids[0, txt_len:, 1] = rows.float()
        ids[0, txt_len:, 2] = cols.float()
        return self.model.pe_embedder(ids)   # [1, 1, S, D/2, 2, 2]

    def qwen_coarse_ropes(self, h, w, f, txt_len, device):
        """As flux_coarse_ropes, but the text ids follow the model's own convention for a
        coarse (h/f) x (w/f) grid: the model places text at max(patch side // 2)."""
        key = ("qcoarse", h, w, f, txt_len, str(device))
        if self._ropes.get("qcoarse", (None,))[0] != key:
            nh, nw = h // f, w // f
            txt_start = float(max(nh // 2, nw // 2))

            def build(rows, cols):
                ids = torch.zeros(1, txt_len + rows.numel(), 3, device=device, dtype=torch.float32)
                txt = torch.arange(txt_start, txt_start + txt_len, device=device)
                ids[0, :txt_len, 0] = txt
                ids[0, :txt_len, 1] = txt
                ids[0, :txt_len, 2] = txt
                ids[0, txt_len:, 1] = rows.float()
                ids[0, txt_len:, 2] = cols.float()
                return self.model.pe_embedder(ids)

            pos = torch.arange(h * w, device=device)
            cpos = torch.arange((h // f) * (w // f), device=device)
            rope_q = build(pos // w // f, pos % w // f)
            rope_kv = build(cpos // (w // f), cpos % (w // f))
            self._ropes["qcoarse"] = (key, (rope_q, rope_kv))
        return self._ropes["qcoarse"][1]

    # ---- ideogram rotary helpers -------------------------------------------

    def ideogram_ropes(self, h, w, f, txt_len, device):
        """(query, coarse-key, true) split-half rotation angles for the Phase I coarse grid.
        Each entry is (cos, sin) of shape [seq, head_dim // 2]; text keeps its own positions."""
        key = (h, w, f, txt_len, str(device))
        if key not in self._ropes:
            self._ropes = {k: v for k, v in self._ropes.items() if k[2] == f}
            import comfy.ldm.ideogram4.model as ideogram4_model
            from comfy.text_encoders.llama import precompute_freqs_cis

            dm = self.model
            head_dim = int(dm.layers[0].attention.head_dim)
            theta = float(dm.rope_theta)
            sections = tuple(dm.mrope_section)
            offset = int(ideogram4_model.IMAGE_POSITION_OFFSET)

            def positions(rows, cols):
                img = torch.stack([torch.zeros_like(rows), rows, cols], dim=-1) + offset
                txt = torch.arange(txt_len, device=device, dtype=torch.long).unsqueeze(1).expand(txt_len, 3)
                return torch.cat([txt.unsqueeze(0), img.unsqueeze(0)], dim=1)   # [1, txt+img, 3]

            pos = torch.arange(h * w, device=device)
            coarse = torch.arange((h // f) * (w // f), device=device)
            true_img = torch.arange(txt_len, device=device, dtype=torch.long).unsqueeze(1).expand(txt_len, 3)
            r = torch.arange(h, device=device).repeat_interleave(w)
            c = torch.arange(w, device=device).repeat(h)
            true_img = torch.cat([true_img.unsqueeze(0),
                                  (torch.stack([torch.zeros_like(r), r, c], dim=-1) + offset).unsqueeze(0)], dim=1)

            def angles(p):
                fc = precompute_freqs_cis(head_dim, p[0].transpose(0, 1), theta,
                                          rope_dims=sections, interleaved_mrope=True, device=device)
                m = ideogram4_model._split_half_rope_matrix(fc)   # [1, L, 1, D/2, 2, 2]
                return m[..., 0, 0], m[..., 1, 0]                 # cos/sin, [1, L, 1, D/2]

            self._ropes[key] = (angles(positions(pos // w // f, pos % w // f)),
                                angles(positions(coarse // (w // f), coarse % (w // f))),
                                angles(true_img))
        return self._ropes[key]


# ---------------------------------------------------------------------------
# rope helpers
# ---------------------------------------------------------------------------

def _rope_apply(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Apply EmbedND-style rotary embedding to x [B, H, S, D] with pe [1, 1, S, D/2, 2, 2].
    Identical math to comfy.ldm.flux.math._apply_rope1."""
    x_ = x.to(pe.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and pe.shape[2] != 1 and x_.shape[2] != pe.shape[2]:
        pe = pe[:, :, :x_.shape[2]]
    out = pe[..., 0] * x_[..., 0]
    out.addcmul_(pe[..., 1], x_[..., 1])
    return out.reshape(*x.shape).type_as(x)


def _identity_pe(tokens: int, head_dim: int, device, dtype) -> torch.Tensor:
    """RoPE table that rotates nothing: cos = 1, sin = 0 as a proper 2x2 identity per pair.
    Applied by the model after an attn1_patch already rotated q/k with the AReA coarse rope."""
    pe = torch.zeros(1, 1, tokens, head_dim // 2, 2, 2, device=device, dtype=torch.float32)
    pe[..., 0, 0] = 1.0
    pe[..., 1, 1] = 1.0
    return pe.to(dtype)


def _rot_split_half(x: torch.Tensor, c: torch.Tensor, s: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """(Un)rotate [B, H, S, D] with split-half RoPE angles ``c, s`` [S, D/2] (Ideogram 4).

    Forward matches the model's split-half rotation (pair = (j, j + D/2),
    out0 = c*x0 - s*x1, out1 = s*x0 + c*x1); ``inverse`` applies the transpose."""
    w = x.shape[-1] // 2
    x0, x1 = x[..., :w], x[..., w:]
    cc = c.to(x.dtype).reshape(1, 1, -1, w)
    ss = s.to(x.dtype).reshape(1, 1, -1, w)
    if inverse:
        return torch.cat([cc * x0 + ss * x1, cc * x1 - ss * x0], dim=-1)
    return torch.cat([cc * x0 - ss * x1, ss * x0 + cc * x1], dim=-1)


# ---------------------------------------------------------------------------
# grid helpers
# ---------------------------------------------------------------------------

def _grid(img_ids: torch.Tensor):
    """Patch grid (h, w) of row-major image ids (AReA FluxBackbone._grid)."""
    rows = (img_ids[0] if img_ids.ndim == 3 else img_ids)[:, 1]
    n = int(rows.numel())
    changed = torch.nonzero(rows != rows[0])
    w = int(changed[0]) if changed.numel() else n
    return n // w, w


def _canonical_grid(img_ids: torch.Tensor, h: int, w: int) -> bool:
    """True when img_ids hold the canonical integer grid (no rope scaling / offsets)."""
    if h * w == 0 or int(img_ids.shape[1]) != h * w:
        return False
    rows = img_ids[0, :, 1].detach().float()
    cols = img_ids[0, :, 2].detach().float()
    ref_r = torch.arange(h, device=rows.device, dtype=rows.dtype).repeat_interleave(w)
    ref_c = torch.arange(w, device=cols.device, dtype=cols.dtype).repeat(h)
    return bool(torch.all(rows == ref_r) and torch.all(cols == ref_c))


def _qwen_canonical_grid(img_ids: torch.Tensor, h: int, w: int) -> bool:
    """True when img_ids match comfy's Qwen-Image canonical grid for one frame at index 0:
    the model centres both axes (linspace(0, n - 1) - n // 2)."""
    if img_ids.ndim != 3 or int(img_ids.shape[0]) != 1 or int(img_ids.shape[1]) != h * w:
        return False
    dev = img_ids.device
    dt = img_ids.dtype
    exp = torch.zeros(1, h * w, 3, device=dev, dtype=torch.float32)
    exp[0, :, 1] = torch.arange(h, device=dev, dtype=dt).repeat_interleave(w).float() - (h // 2)
    exp[0, :, 2] = torch.arange(w, device=dev, dtype=dt).repeat(h).float() - (w // 2)
    return bool(torch.equal(img_ids.detach().float(), exp))


def _build_fine_or_none(state, tokens, h, w):
    """The Phase II layout of one step, or None (plain attention) if it fails."""
    try:
        return state.fine_layout(tokens, h, w, state.step, state.size, state.layout_txt_len)
    except Exception:
        logger.exception("AReA fine layout failed; falling back to plain attention")
        return None


# ---------------------------------------------------------------------------
# FLUX.1 hooks
# ---------------------------------------------------------------------------

def make_flux_wrapper(state: AReAState):
    """wrappers[WrappersMP.DIFFUSION_MODEL]: runs before every Flux.forward.
    Same calling convention as comfy's own wrappers (see nodes_easycache.py):
    wrapper(executor, *args, **kwargs), Flux args = (x, timestep, context, y, guidance,
    ref_latents, control, transformer_options); the Flux module is executor.class_obj."""

    def wrapper(executor, *args, **kwargs):
        state.reset_eval()
        try:
            flux_model = executor.class_obj
            if flux_model is None:
                return executor(*args, **kwargs)
            state.model = flux_model
            x = args[0] if len(args) > 0 else kwargs.get("x")
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            ref_latents = args[5] if len(args) > 5 else kwargs.get("ref_latents")
            topts = args[7] if len(args) > 7 else kwargs.get("transformer_options", {})
            if x is None or timestep is None or not isinstance(topts, dict):
                return executor(*args, **kwargs)
            # reference latents (kontext / fill / redux) change the token grid: keep plain attention
            if ref_latents is not None or context is None:
                return executor(*args, **kwargs)
            located = state.locate_step(timestep, topts)
            if located is None:
                return executor(*args, **kwargs)
            state.step, state.data["sigmas"] = located
            state.txt_len = state.layout_txt_len = int(context.shape[1])
        except Exception:
            state.reset_eval()
            logger.exception("AReA step wrapper failed; falling back to plain attention")
        return executor(*args, **kwargs)

    return wrapper


def make_flux_post_input(state: AReAState):
    """patches["post_input"]: after img/txt embeddings, before the blocks."""

    def patch(inputs):
        try:
            if state.step is None or state.model is None:
                return inputs
            img = inputs["img"]
            img_ids = inputs["img_ids"]
            h, w = _grid(img_ids)
            if h * w <= 0 or img.shape[1] != h * w or not _canonical_grid(img_ids, h, w):
                return inputs
            sched = state.schedule(h, w, max(int(state.data["sigmas"].numel()) - 1, 1), state.txt_len)
            if sched is None:
                return inputs
            state.phase, state.size = sched.at(state.step)
            state.h, state.w = h, w
            cfg = state.config
            device = img.device

            if state.phase == "coarse":
                f = state.size
                if (h % f) or (w % f):
                    state.phase = None
                    return inputs
                state.data["rope_q"], state.data["rope_kv"] = state.flux_coarse_ropes(
                    h, w, f, state.txt_len, device)
            elif state.phase == "fine":
                # same role as AReA's x_embedder forward hook: build the fine layout from the
                # embedded image tokens (DWT detail map -> adaptive windows -> Halo + bridge)
                layout = _build_fine_or_none(state, img, h, w)
                if layout is None:
                    state.phase = None
                    return inputs
                state.data["layout"] = layout
        except Exception:
            state.reset_eval()
            logger.exception("AReA post_input patch failed; falling back to plain attention")
        return inputs

    return patch


def make_flux_attn1_patch(state: AReAState):
    """patches["attn1_patch"]: q/k/v are [B, heads, S, D], sequence = [text | image]."""

    def patch(q, k, v, pe=None, attn_mask=None, extra_options={}):
        phase = state.phase
        if phase is None or attn_mask is not None or pe is None:
            return {}
        try:
            sl = extra_options.get("img_slice")
            t = int(sl[0]) if sl else state.txt_len
            b, heads, s, dim = q.shape
            cfg = state.config

            if phase == "coarse":
                f = state.size
                if s - t != state.h * state.w or t <= 0:
                    return {}
                q_s = q.transpose(1, 2)          # [B, S, H, D]
                k_s = k.transpose(1, 2)
                v_s = v.transpose(1, 2)
                k_p, v_p = area_attention.pool_keys_values(
                    k_s, v_s, t, state.h, state.w, f, cfg.pooling,
                    seed=cfg.seed, step=state.step,
                    block=int(extra_options.get("block_index", 0)))
                # AReA custom RoPE: queries quantised to their block's coarse position, pooled
                # keys on the coarse grid. pe=None => the block's attention applies no extra rope.
                # _rope_apply works on [B, H, S, D]; q keeps its sequence, k/v are the pooled ones.
                q_r = _rope_apply(q, state.data["rope_q"])
                k_r = _rope_apply(k_p.transpose(1, 2).contiguous(), state.data["rope_kv"])
                return {
                    "q": q_r,
                    "k": k_r,
                    "v": v_p.transpose(1, 2).contiguous(),
                    "pe": None,
                }

            # fine phase
            layout = state.data.get("layout")
            if layout is None or s - t != layout.h * layout.w:
                return {}
            q_r = _rope_apply(q, pe)             # true grid RoPE == the model's own pe
            k_r = _rope_apply(k, pe)
            q_r, k_r = q_r.transpose(1, 2), k_r.transpose(1, 2)
            v_s = v.transpose(1, 2)
            out_txt, out_img = area_attention.windowed_attention(
                q_r[:, :t], k_r[:, :t], v_s[:, :t],
                q_r[:, t:], k_r[:, t:], v_s[:, t:],
                k.transpose(1, 2)[:, t:],        # unrotated keys rank the bridge tokens
                layout)
            out = torch.cat([out_txt, out_img], dim=1)      # [B, S, H, D]
            state.data.setdefault("phase2_out", {})[_block_key(extra_options)] = out.reshape(b, s, heads * dim).to(q.dtype)
            # make the block's own attention nearly free (linear cost, O(S*D)); the output
            # patch replaces its result with the stashed windowed attention
            return {"q": q, "k": k[:, :, :1], "v": v[:, :, :1], "pe": None}
        except Exception:
            state.reset_eval()
            logger.exception("AReA attn1_patch failed; falling back to plain attention")
            return {}

    return patch


def make_flux_attn1_output_patch(state: AReAState):
    """patches["attn1_output_patch"]: attn is [B, S, H*D]."""

    def patch(attn, extra_options):
        if state.phase != "fine":
            return attn
        out = state.data["phase2_out"].pop(_block_key(extra_options), None)
        return attn if out is None else out

    return patch


# ---------------------------------------------------------------------------
# Qwen-Image hooks
# ---------------------------------------------------------------------------

def make_qwen_wrapper(state: AReAState):
    """Qwen-Image forward args: (x, timestep, context, attention_mask, ref_latents,
    additional_t_cond, transformer_options). x is [B, C, T, H, W]."""

    def wrapper(executor, *args, **kwargs):
        state.reset_eval()
        try:
            dm = executor.class_obj
            if dm is None:
                return executor(*args, **kwargs)
            state.model = dm
            x = args[0] if len(args) > 0 else kwargs.get("x")
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            attention_mask = args[3] if len(args) > 3 else kwargs.get("attention_mask")
            ref_latents = args[4] if len(args) > 4 else kwargs.get("ref_latents")
            topts = args[6] if len(args) > 6 else kwargs.get("transformer_options", {})
            if (x is None or timestep is None or context is None or not isinstance(topts, dict)
                    or x.dim() != 5 or int(x.shape[2]) != 1):
                return executor(*args, **kwargs)
            # edit / fill style reference latents and padded multi-prompt masks: plain attention
            if ref_latents is not None or (attention_mask is not None
                                           and not bool((attention_mask == 1).all())):
                return executor(*args, **kwargs)
            state.data["mask_trivial"] = True   # mask absent or all-ones: no extra attention constraint
            located = state.locate_step(timestep, topts)
            if located is None:
                return executor(*args, **kwargs)
            state.step, state.data["sigmas"] = located
            state.txt_len = state.layout_txt_len = int(context.shape[1])
            ps = max(1, int(getattr(dm, "patch_size", 2)))
            state.data["grid"] = ((int(x.shape[-2]) + ps // 2) // ps, (int(x.shape[-1]) + ps // 2) // ps)
        except Exception:
            state.reset_eval()
            logger.exception("AReA step wrapper failed; falling back to plain attention")
        return executor(*args, **kwargs)

    return wrapper


def make_qwen_post_input(state: AReAState):
    def patch(inputs):
        try:
            if state.step is None or state.model is None:
                return inputs
            img = inputs["img"]
            h, w = state.data["grid"]
            if h * w <= 0 or int(img.shape[1]) != h * w or not _qwen_canonical_grid(inputs["img_ids"], h, w):
                return inputs
            sched = state.schedule(h, w, max(int(state.data["sigmas"].numel()) - 1, 1), state.txt_len)
            if sched is None:
                return inputs
            state.phase, state.size = sched.at(state.step)
            state.h, state.w = h, w
            device = img.device

            if state.phase == "coarse":
                f = state.size
                if (h % f) or (w % f):
                    state.phase = None
                    return inputs
                state.data["rope_q"], state.data["rope_kv"] = state.qwen_coarse_ropes(
                    h, w, f, state.txt_len, device)
            elif state.phase == "fine":
                layout = _build_fine_or_none(state, img, h, w)
                if layout is None:
                    state.phase = None
                    return inputs
                state.data["layout"] = layout
        except Exception:
            state.reset_eval()
            logger.exception("AReA post_input patch failed; falling back to plain attention")
        return inputs

    return patch


def make_qwen_attn1_patch(state: AReAState):
    """patches["attn1_patch"] on the comfy Qwen-Image attention: q/k/v [B, heads, S, D],
    taken BEFORE the model applies RoPE. comfy's Qwen-Image applies RoPE unconditionally
    after this patch, so Phase I returns pre-rotated q/k with an identity RoPE table."""

    def patch(q, k, v, pe=None, attn_mask=None, extra_options={}):
        phase = state.phase
        if phase is None or pe is None:
            return {}
        try:
            sl = extra_options.get("img_slice")
            t = int(sl[0]) if sl else state.txt_len
            b, heads, s, dim = q.shape
            cfg = state.config

            if phase == "coarse":
                f = state.size
                if s - t != state.h * state.w or t <= 0:
                    return {}
                q_s = q.transpose(1, 2)
                k_s = k.transpose(1, 2)
                v_s = v.transpose(1, 2)
                k_p, v_p = area_attention.pool_keys_values(
                    k_s, v_s, t, state.h, state.w, f, cfg.pooling,
                    seed=cfg.seed, step=state.step,
                    block=int(extra_options.get("block_index", 0)))
                q_r = _rope_apply(q, state.data["rope_q"])
                k_r = _rope_apply(k_p.transpose(1, 2).contiguous(), state.data["rope_kv"])
                return {"q": q_r, "k": k_r, "v": v_p.transpose(1, 2).contiguous(),
                        "pe": _identity_pe(s, dim, q.device, q.dtype)}

            # fine phase: the attention override runs the windowed attention; it needs this
            # block's unrotated keys for the bridge ranking (stashed, reference only)
            state.data["unrot_k"] = k
            return {}
        except Exception:
            state.reset_eval()
            logger.exception("AReA qwen attn1_patch failed; falling back to plain attention")
            return {}

    return patch


def make_qwen_attention_override(state: AReAState):
    """optimized_attention_override for Qwen-Image Phase II (no attn1_output_patch on this
    model): replaces the block's joint attention with the AReA windowed attention. q/k
    arrive already rotated with the true grid RoPE; v is raw.

    Registered through ModelPatcher.set_model_optimized_attention, whose internal wrapper
    consumes the leading attention-function argument itself — the signature here is
    (q, k, v, heads, ...). The plain-attention fallback goes through comfy's active
    optimized_attention_masked (the caller's wrap_attn guard ``_inside_attn_wrapper``
    prevents re-entering this override)."""

    def override(q, k, v, heads, mask=None, **kwargs):
        import comfy.ldm.modules.attention

        def plain():
            return comfy.ldm.modules.attention.optimized_attention_masked(q, k, v, heads, mask, **kwargs)

        try:
            if state.phase != "fine" or (mask is not None and not state.data.get("mask_trivial")):
                return plain()
            layout = state.data.get("layout")
            k_unrot = state.data.pop("unrot_k", None)
            t = state.txt_len
            b, hh, s, dim = q.shape
            if (layout is None or k_unrot is None or k_unrot.shape != q.shape
                    or t <= 0 or s - t != state.h * state.w):
                return plain()
            q_r = q.transpose(1, 2)              # [B, S, H, D], rotated true grid
            k_r = k.transpose(1, 2)
            v_s = v.transpose(1, 2)
            out_txt, out_img = area_attention.windowed_attention(
                q_r[:, :t], k_r[:, :t], v_s[:, :t],
                q_r[:, t:], k_r[:, t:], v_s[:, t:],
                k_unrot.transpose(1, 2)[:, t:],
                layout)
            out = torch.cat([out_txt, out_img], dim=1)
            return out.reshape(b, s, heads * dim).to(q.dtype)
        except Exception:
            state.reset_eval()
            logger.exception("AReA qwen attention override failed; falling back to plain attention")
            return plain()

    return override


# ---------------------------------------------------------------------------
# Ideogram 4 hooks
# ---------------------------------------------------------------------------

def make_ideogram_wrapper(state: AReAState):
    """Ideogram 4 forward args: (x, timesteps, context, attention_mask, transformer_options).
    x is [B, C, gh, gw] and gh/gw ARE the patch grid (the 2x2 patchify splits channels).
    The unconditional pass runs with context=None (image-only sequence)."""

    def wrapper(executor, *args, **kwargs):
        state.reset_eval()
        try:
            dm = executor.class_obj
            if dm is None:
                return executor(*args, **kwargs)
            state.model = dm
            x = args[0] if len(args) > 0 else kwargs.get("x")
            timestep = args[1] if len(args) > 1 else kwargs.get("timesteps")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            attention_mask = args[3] if len(args) > 3 else kwargs.get("attention_mask")
            topts = args[4] if len(args) > 4 else kwargs.get("transformer_options", {})
            if x is None or timestep is None or not isinstance(topts, dict) or x.dim() != 4:
                return executor(*args, **kwargs)
            if attention_mask is not None and not bool((attention_mask == 1).all()):
                return executor(*args, **kwargs)   # padded / multi-prompt batch: plain attention
            state.data["mask_trivial"] = True
            located = state.locate_step(timestep, topts)
            if located is None:
                return executor(*args, **kwargs)
            state.step, state.data["sigmas"] = located
            h, w = int(x.shape[-2]), int(x.shape[-1])
            txt_len = int(context.shape[1]) if context is not None else 0
            if context is not None:
                state.last_txt_len = txt_len
            # share the conditional pass's layout, as the upstream backbone does
            state.layout_txt_len = txt_len if context is not None else state.last_txt_len
            state.txt_len = txt_len
            steps_total = max(int(state.data["sigmas"].numel()) - 1, 1)
            sched = state.schedule(h, w, steps_total, state.layout_txt_len)
            if sched is None:
                return executor(*args, **kwargs)
            phase, size = sched.at(state.step)
            is_last = phase == "fine" and state.step == steps_total - 1
            if is_last:
                # the last step carries a large share of the denoising: keep every start window
                # whole (no DWT splits), or small windows settle their brightness independently
                size = sched.windows[0]
            state.phase, state.size = phase, size
            state.h, state.w = h, w
            state._is_last = is_last
            if phase == "coarse":
                state.data["ropes"] = state.ideogram_ropes(h, w, size, txt_len, x.device)
            elif phase == "fine":
                state.data["rope_true"] = state.ideogram_ropes(h, w, size, txt_len, x.device)[2]
                tokens = x.permute(0, 2, 3, 1).reshape(x.shape[0], h * w, x.shape[1]).to(x.dtype)
                layout = _build_fine_or_none(state, tokens, h, w)
                if layout is None:
                    state.phase = None
                    return executor(*args, **kwargs)
                state.data["layout"] = layout
        except Exception:
            state.reset_eval()
            logger.exception("AReA step wrapper failed; falling back to plain attention")
        return executor(*args, **kwargs)

    return wrapper


def make_ideogram_attention_override(state: AReAState):
    """optimized_attention_override for Ideogram 4 (both phases). q/k arrive already rotated
    with the model's split-half MRoPE at the true positions, v is raw; block identity comes
    from a per-forward call counter (the model exposes no block_index). Signature note: the
    ModelPatcher.set_model_optimized_attention wrapper consumes the leading attention-function
    argument itself, so this override starts directly at (q, k, v, heads, ...)."""

    def override(q, k, v, heads, mask=None, **kwargs):
        import comfy.ldm.modules.attention

        def plain():
            return comfy.ldm.modules.attention.optimized_attention_masked(q, k, v, heads, mask, **kwargs)

        try:
            phase = state.phase
            if phase is None or (mask is not None and not state.data.get("mask_trivial")):
                return plain()
            block = state.block_counter
            state.block_counter += 1
            b, hh, s, dim = q.shape
            t = state.txt_len
            if s - t != state.h * state.w:
                return plain()
            cfg = state.config

            if phase == "coarse":
                ropes = state.data.get("ropes")
                f = state.size
                if ropes is None or (state.h % f) or (state.w % f) or t < 0:
                    return plain()
                (qc, qs), (kc, ks), (tc, ts) = ropes
                # recover unrotated q/k with the inverse of the true rotation, then apply the
                # AReA coarse rotation (queries on block coords, pooled keys on the coarse grid)
                q_un = _rot_split_half(q, tc, ts, inverse=True)
                q_r = _rot_split_half(q_un, qc, qs)
                k_un = _rot_split_half(k, tc, ts, inverse=True)
                k_s = k_un.transpose(1, 2)
                v_s = v.transpose(1, 2)
                k_p, v_p = area_attention.pool_keys_values(
                    k_s, v_s, t, state.h, state.w, f, cfg.pooling,
                    seed=cfg.seed, step=state.step, block=block)
                k_r = _rot_split_half(k_p.transpose(1, 2).contiguous(), kc, ks)
                # route through comfy's own attention backend: Phase I keeps full-resolution
                # queries, and a raw SDPA call can fall back to the math kernel and materialise
                # the whole score matrix (OOM at 4K); the stock model runs this same shape
                # through this backend every step. skip_reshape: q/k/v are [B, H, S, D].
                out = comfy.ldm.modules.attention.optimized_attention(
                    q_r, k_r, v_p.transpose(1, 2), heads, skip_reshape=True)
                return out.to(q.dtype)

            # fine phase
            layout = state.data.get("layout")
            true_pair = state.data.get("rope_true")
            if layout is None or true_pair is None:
                return plain()
            q_s = q.transpose(1, 2)
            k_s = k.transpose(1, 2)
            v_s = v.transpose(1, 2)
            k_unrot = _rot_split_half(k, true_pair[0], true_pair[1], inverse=True).transpose(1, 2)
            out_txt, out_img = area_attention.windowed_attention(
                q_s[:, :t], k_s[:, :t], v_s[:, :t],
                q_s[:, t:], k_s[:, t:], v_s[:, t:],
                k_unrot[:, t:],
                layout)
            out = torch.cat([out_txt, out_img], dim=1)
            return out.reshape(b, s, heads * dim).to(q.dtype)
        except Exception:
            state.reset_eval()
            logger.exception("AReA ideogram attention override failed; falling back to plain attention")
            return plain()

    return override


# ---------------------------------------------------------------------------
# dispatch helpers
# ---------------------------------------------------------------------------

def _block_key(extra_options):
    return (extra_options.get("block_type"), extra_options.get("block_index"))


def make_wrapper(state: AReAState):
    if state.backend == "flux":
        return make_flux_wrapper(state)
    if state.backend == "qwen_image":
        return make_qwen_wrapper(state)
    return make_ideogram_wrapper(state)
