"""ComfyUI-AReA: AReA (Adaptive Receptive-Field Attention) for native FLUX.1, Qwen-Image
and Ideogram 4.

Based on https://github.com/KooroshRH/AReA (Apache-2.0). Runs on the native
ComfyUI inference pipeline via ModelPatcher extension points only — no diffusers,
no ComfyUI source modifications.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
