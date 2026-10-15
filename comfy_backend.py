"""ComfyUI native-pipeline adapter for AReA (FLUX.1).

Nothing here touches ComfyUI source: everything hooks through the official
ModelPatcher extension points of comfy/ldm/flux:

- ``wrappers[diffusion_model]``  -> locates the current denoising step from the
  model timestep and ``transformer_options["sample_sigmas"]`` (same pattern as
  comfy/context_windows.py), keeps a reference to the Flux module.
- ``patches["post_input"]``      -> runs right after the img/txt embeddings:
  builds the per-step schedule state, the Phase I coarse RoPE tables and the
  Phase II fine layout (DWT detail map, adaptive windows, Halo, bridge).
- ``patches["attn1_patch"]``     -> Phase I: pools K/V to one token per f x f
  block and applies AReA's custom RoPE (queries quantised to block coords,
  keys on the coarse grid) with pe returned as None so the block's own
  attention runs on the pooled KV. Phase II: computes the windowed attention
  (windows + Halo + global-KV bridge) and stashes the result.
- ``patches["attn1_output_patch"]`` -> Phase II only: swaps the stashed
  windowed-attention result for the block's (deliberately trivialised)
  attention output.

RoPE math mirrors comfy.ldm.flux.math._apply_rope1 exactly (cos/sin packed in
the EmbedND [.., D/2, 2, 2] layout) and position tables come from the model's
own ``pe_embedder`` — so AReA needs no RoPE extrapolation and no weight
changes, matching the upstream diffusers implementation.
"""

import logging

import torch
import torch.nn.functional as F

from . import area
from .area import attention as area_attention
from .area import schedule as area_schedule
from .area import tiling as area_tiling

logger = logging.getLogger("comfy.custom_nodes.ComfyUI-AReA")

NATIVE_PATCHES = 64          # FLUX.1 native training extent: 1024 px / 16 px per patch
DEFAULT_PHASE1_STEPS = 8     # of 28


class AReAConfig:
    def __init__(self, pooling="subsample", phase1_steps=DEFAULT_PHASE1_STEPS,
                 halo=8, bridge_ratio=0.5, seed=0, native_patches=NATIVE_PATCHES):
        if pooling not in area_attention.POOLING:
            raise ValueError(f"pooling must be one of {area_attention.POOLING}, got {pooling!r}")
        self.pooling = pooling
        self.phase1_steps = int(phase1_steps)
        self.halo = int(halo)
        self.bridge_ratio = float(bridge_ratio)
        self.seed = int(seed)
        self.native_patches = int(native_patches)


class AReAState:
    """Per-step denoising state shared by every patch of one patched model."""

    def __init__(self, config: AReAConfig):
        self.config = config
        self.phase = None          # None: plain attention, "coarse": Phase I, "fine": Phase II
        self.step = None
        self.size = 0              # pooling factor f (Phase I) / window side w (Phase II)
        self.h = 0
        self.w = 0
        self.txt_len = 0
        self.flux_model = None
        self.data = {}
        self._schedule = None
        self._schedule_key = None
        self._ropes = {}

    def reset_eval(self):
        """Called once per diffusion-model evaluation (one sampler step, both CFG branches)."""
        self.phase = None
        self.data = {"phase2_out": {}}

    def schedule(self, h, w, steps, txt_len):
        if max(h, w) <= self.config.native_patches:
            return None
        phase1 = self.config.phase1_steps
        key = (h, w, steps, phase1, txt_len)
        if key != self._schedule_key:
            self._schedule = area_schedule.build_schedule(h, w, steps, phase1,
                                                          self.config.native_patches, txt_len=txt_len)
            self._schedule_key = key
        return self._schedule

    # ---- rotary embedding helpers -----------------------------------------

    def _rope(self, txt_len, rows, cols, device):
        ids = torch.zeros(1, txt_len + rows.numel(), 3, device=device, dtype=torch.float32)
        ids[0, txt_len:, 1] = rows.float()
        ids[0, txt_len:, 2] = cols.float()
        return self.flux_model.pe_embedder(ids)   # [1, 1, S, D/2, 2, 2]

    def coarse_ropes(self, h, w, f, txt_len, device):
        """Queries keep full resolution and share the coarse position of their block; pooled keys
        sit on the (h/f) x (w/f) grid, so both are measured in the same units."""
        key = ("coarse", h, w, f, txt_len, str(device))
        if self._ropes.get("coarse", (None,))[0] != key:
            pos = torch.arange(h * w, device=device)
            cpos = torch.arange((h // f) * (w // f), device=device)
            rope_q = self._rope(txt_len, pos // w // f, pos % w // f, device)
            rope_kv = self._rope(txt_len, cpos // (w // f), cpos % (w // f), device)
            self._ropes["coarse"] = (key, (rope_q, rope_kv))
        return self._ropes["coarse"][1]


def _rope_apply(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Apply EmbedND-style rotary embedding to x [B, H, S, D] with pe [1, 1, S, D/2, 2, 2].
    Identical math to comfy.ldm.flux.math._apply_rope1."""
    x_ = x.to(pe.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and pe.shape[2] != 1 and x_.shape[2] != pe.shape[2]:
        pe = pe[:, :, :x_.shape[2]]
    out = pe[..., 0] * x_[..., 0]
    out.addcmul_(pe[..., 1], x_[..., 1])
    return out.reshape(*x.shape).type_as(x)


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


def _block_key(extra_options):
    return (extra_options.get("block_type"), extra_options.get("block_index"))


# ---------------------------------------------------------------------------
# The four hooks registered on the ModelPatcher
# ---------------------------------------------------------------------------

def make_diffusion_model_wrapper(state: AReAState):
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
            state.flux_model = flux_model
            x = args[0] if len(args) > 0 else kwargs.get("x")
            timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            ref_latents = args[5] if len(args) > 5 else kwargs.get("ref_latents")
            topts = args[7] if len(args) > 7 else kwargs.get("transformer_options", {})
            if x is None or timestep is None or not isinstance(topts, dict):
                return executor(*args, **kwargs)
            sigmas = topts.get("sample_sigmas", None)
            # reference latents (kontext / fill / redux) change the token grid: keep plain attention
            if sigmas is None or ref_latents is not None or context is None:
                return executor(*args, **kwargs)

            sigma_now = float(timestep.detach().reshape(-1)[0])
            sig = sigmas.detach().float().cpu()
            diffs = (sig - sigma_now).abs()
            idx = int(diffs.argmin())
            if float(diffs[idx]) > 1e-3 * max(1.0, sigma_now) + 1e-6:
                return executor(*args, **kwargs)   # not on the registered schedule: plain attention

            state.step = idx
            state.txt_len = int(context.shape[1])
            state.data["sigmas"] = sig
        except Exception:
            state.reset_eval()
            logger.exception("AReA step wrapper failed; falling back to plain attention")
        return executor(*args, **kwargs)

    return wrapper


def make_post_input_patch(state: AReAState):
    """patches["post_input"]: after img/txt embeddings, before the blocks."""

    def patch(inputs):
        try:
            if state.step is None or state.flux_model is None:
                return inputs
            img = inputs["img"]
            txt = inputs["txt"]
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
                state.data["rope_q"], state.data["rope_kv"] = state.coarse_ropes(
                    h, w, f, state.txt_len, device)
            elif state.phase == "fine":
                # same role as AReA's x_embedder forward hook: build the fine layout from the
                # embedded image tokens (DWT detail map -> adaptive windows -> Halo + bridge)
                w_min = min(cfg.native_patches, max(h, w))
                energy = area_tiling.detail_map(img, h, w)
                windows = area_tiling.adaptive_tiling(
                    energy, h, w, state.size, w_min, state.txt_len,
                    rng=area_tiling.step_rng(cfg.seed, state.step, 0x5EA1))
                registers = (max(1, int(round(state.size * cfg.bridge_ratio))) ** 2
                             if cfg.bridge_ratio > 0 else 0)
                # never cost more than a full-attention step: cut bridge tokens, then the Halo
                registers, halo = area_attention.fit_extra_keys(
                    windows, h, w, state.txt_len, registers, cfg.halo)
                state.data["layout"] = area_attention.build_fine_layout(
                    windows, h, w, halo, registers, cfg.seed, state.step, img.dtype, device)
        except Exception:
            state.reset_eval()
            logger.exception("AReA post_input patch failed; falling back to plain attention")
        return inputs

    return patch


def make_attn1_patch(state: AReAState):
    """patches["attn1_patch"]: q/k/v are [B, heads, S, D], sequence = [text | image]."""

    def patch(q, k, v, pe=None, attn_mask=None, extra_options={}):
        phase = state.phase
        if phase is None or attn_mask is not None or pe is None:
            return {}
        try:
            block = _block_key(extra_options)
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
                    seed=cfg.seed, step=state.step, block=int(extra_options.get("block_index", 0)))
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
            state.data["phase2_out"][block] = out.reshape(b, s, heads * dim).to(q.dtype)
            # make the block's own attention nearly free (linear cost, O(S*D)); the output
            # patch replaces its result with the stashed windowed attention
            return {"q": q, "k": k[:, :, :1], "v": v[:, :, :1], "pe": None}
        except Exception:
            state.reset_eval()
            logger.exception("AReA attn1_patch failed; falling back to plain attention")
            return {}

    return patch


def make_attn1_output_patch(state: AReAState):
    """patches["attn1_output_patch"]: attn is [B, S, H*D]."""

    def patch(attn, extra_options):
        if state.phase != "fine":
            return attn
        out = state.data["phase2_out"].pop(_block_key(extra_options), None)
        return attn if out is None else out

    return patch
