"""AReA model-independent core, ported from https://github.com/KooroshRH/AReA (Apache-2.0).

schedule.py / tiling.py / bridge.py are copied verbatim.
attention.py is adapted: the diffusers attention dispatch is replaced with
torch scaled_dot_product_attention on [B, S, H, D] tensors.
"""
