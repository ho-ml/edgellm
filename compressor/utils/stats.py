import torch

from typing import *
from dataclasses import dataclass
from compressor.nn import AttentionConfig

__all__ = [
    "ActStats", "get_act_max", "get_weight_max",
    "reduce_for_gqa", "expand_for_gqa", "reduce_for_rope"
]

@dataclass
class ActStats:
    """
    Per-channel activation statistics
    """
    min_val: Optional[torch.Tensor] = None
    max_val: Optional[torch.Tensor] = None
    num_samples: int = 0

    def update(self, x: torch.Tensor):
        """
        Update statistics with new activation tensor
        """
        hidden_dim = x.shape[-1]
        flat = x.view(-1, hidden_dim)

        batch_min = flat.min(dim=0)[0]
        batch_max = flat.max(dim=0)[0]

        if self.min_val is None:
            self.min_val = batch_min
            self.max_val = batch_max
        else:
            self.min_val = torch.minimum(self.min_val, batch_min)
            self.max_val = torch.maximum(self.max_val, batch_max)

        self.num_samples += flat.shape[0]

    @property
    def absmax(self):
        """
        Per-channel absolute max
        """
        if self.min_val is None:
            return None

        return torch.maximum(self.min_val.abs(), self.max_val.abs())

def get_act_max(
    stats: ActStats,
    attn_config: AttentionConfig = None,
    is_kv: bool = False,
    gqa: bool = False
):
    """
    Get activation absmax with RoPE and GQA handling
    """
    amax = stats.absmax.clamp(min=1e-8)

    # apply RoPE handling with correct head count
    if attn_config is not None and attn_config.with_rope:
        if is_kv and attn_config.num_kv_groups > 1:
            # K/V projections use num_kv_heads
            num_heads = attn_config.num_heads // attn_config.num_kv_groups
        else:
            # Q projection uses num_heads
            num_heads = attn_config.num_heads
        amax = reduce_for_rope(amax, num_heads)

    # apply GQA transformation
    if gqa and attn_config is not None and attn_config.num_kv_groups > 1:
        if is_kv:
            amax = expand_for_gqa(amax, attn_config)
        else:
            amax = reduce_for_gqa(amax, attn_config)

    return amax


def get_weight_max(
    weight: torch.Tensor,
    dim: int = 0,
    attn_config: AttentionConfig = None,
    reduce_gqa: bool = False
):
    """
    Get weight absmax
    """
    wmax = weight.abs().amax(dim=dim)

    if reduce_gqa and attn_config is not None:
        wmax = reduce_for_gqa(wmax, attn_config)

    return wmax


def reduce_for_gqa(tensor: torch.Tensor, attn_config: AttentionConfig):
    """
    Reduce tensor from Q dim to KV dim
    """
    if attn_config is None or attn_config.num_kv_groups <= 1:
        return tensor

    num_heads = attn_config.num_heads
    head_dim = attn_config.head_size
    num_kv_groups = attn_config.num_kv_groups
    num_kv_heads = num_heads // num_kv_groups

    return tensor.view(num_kv_heads, num_kv_groups, head_dim)[:, 0, :].reshape(-1)


def expand_for_gqa(tensor: torch.Tensor, attn_config: AttentionConfig):
    """
    Expand tensor from KV dim to Q dim
    """
    if attn_config is None or attn_config.num_kv_groups <= 1:
        return tensor

    num_heads = attn_config.num_heads
    head_dim = attn_config.head_size
    num_kv_groups = attn_config.num_kv_groups
    num_kv_heads = num_heads // num_kv_groups

    return tensor.view(num_kv_heads, 1, head_dim).expand(
        num_kv_heads, num_kv_groups, head_dim
    ).reshape(-1)


def reduce_for_rope(tensor: torch.Tensor, num_heads: int):
    """
    Handle RoPE dimension pairing
    """
    head_dim = tensor.numel() // num_heads
    reshaped = tensor.view(num_heads, head_dim)
    
    # pair adjacent dimensions (sin/cos pattern)
    paired = reshaped.view(num_heads, 2, head_dim // 2)
    max_pairs = paired.amax(dim=1, keepdim=True)
    
    return max_pairs.expand(num_heads, 2, head_dim // 2).reshape(-1)