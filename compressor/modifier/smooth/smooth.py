import torch
import torch.nn as nn

from typing import *
from compressor.nn import AttentionConfig
from compressor.modifier.smooth.scale import scale_up, scale_down
from compressor.utils import expand_for_gqa

__all__ = ["smooth_attn", "smooth_quant"]

def smooth_attn(
    q_proj: nn.Module,
    k_proj: nn.Module,
    scale: torch.Tensor,
    attn_config: AttentionConfig = None
):
    """
    SmoothAttn: balance Q and K projections

    Q' = Q * scale_q, K' = K / scale
    """
    # expand to Q dimension
    scale_q = expand_for_gqa(scale, attn_config)

    # smooth up Q
    scale_up(q_proj.weight, scale_q, dim=0)
    if q_proj.bias is not None:
        scale_up(q_proj.bias, scale_q, dim=0)

    # smooth down K
    scale_down(k_proj.weight, scale, dim=0)
    if k_proj.bias is not None:
        scale_down(k_proj.bias, scale, dim=0)

def smooth_quant(
    prev: nn.Module, balance: List[nn.Module] | nn.Module, scale: torch.Tensor
):
    """
    SmoothQuant: apply smooth to prev & compensate to balance

    prev' = prev / S, balance' = balance * S
    """
    if isinstance(balance, nn.Module):
        balance = [balance]

    # GQA: repeat scale to match dimension
    upscale = scale
    prev_out_dim, bal_in_dim = prev.weight.shape[0], balance[0].weight.shape[1]
    if prev_out_dim != bal_in_dim:
        upscale = scale.repeat_interleave(bal_in_dim // prev_out_dim)
    
    # smooth down prev
    scale_down(prev.weight, scale, dim=0)
    if hasattr(prev, "bias") and prev.bias is not None:
        scale_down(prev.bias, scale, dim=0)

    # smooth up balance
    for proj in balance:
        scale_up(proj.weight, upscale, dim=1)