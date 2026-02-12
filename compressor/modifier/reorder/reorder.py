import torch
import torch.nn as nn

from typing import *
from compressor.utils import ActStats

__all__ = [
    "compute_index",
    "reorder",
    "compute_head_index"
]

def compute_index(
    stats: ActStats,
    num_channels: int,
    num_heads: int = 1,
    num_repeats: int = 1
):
    """
    Compute channel reorder index from activation statistics
    """
    # compute per-channel importance metric
    metric = stats.absmax.to(torch.float32).view(num_channels)

    # per-head reordering
    if num_heads > 1:
        head_channels = num_channels // num_heads

        # handle GQA
        if num_repeats > 1:
            num_unique_heads = num_heads // num_repeats
            metric = metric.view(num_unique_heads, num_repeats, head_channels)
            
            # get max for each heads
            metric = metric.amax(dim=1, keepdim=True)
            rank = metric.argsort(dim=-1).expand(num_unique_heads, num_repeats, -1)
            rank = rank.reshape(num_heads, -1)
        else:
            rank = metric.view(num_heads, head_channels).argsort(dim=-1)

        # head offset
        delta = torch.arange(0, num_channels, head_channels, dtype=torch.long, device=rank.device)
        rank = rank + delta.view(num_heads, 1)

        return rank.view(-1)
    
    # per-channel
    else:
        return metric.argsort()


def reorder(
    input_projs: List[nn.Linear] | nn.Linear,
    index: torch.Tensor,
    output_projs: List[nn.Linear] | nn.Linear | None = None,
    output_index: torch.Tensor | None = None
):
    """
    Reorder weight channels
    """
    if isinstance(input_projs, nn.Module):
        input_projs = [input_projs]
    device = input_projs[0].weight.device

    # reorder input channels (dim=1)
    for proj in input_projs:
        proj.weight.data = proj.weight.data.index_select(1, index.to(device))

    if output_projs is not None:
        if isinstance(output_projs, nn.Module):
            output_projs = [output_projs]
        
        # get output index
        out_idx = output_index if output_index is not None else index
        
        # reorder output channels (dim=0)
        for proj in output_projs:
            proj.weight.data = proj.weight.data.index_select(0, out_idx.to(device))
            if proj.bias is not None:
                proj.bias.data = proj.bias.data[out_idx.to(proj.bias.device)]


def compute_head_index(
    o_index: torch.Tensor,
    num_heads: int,
    num_repeats: int
):
    """
    Convert o_proj index to v_proj index for GQA
    """
    num_channels = o_index.numel()
    head_channels = num_channels // num_heads
    kv_heads = num_heads // num_repeats
    device = o_index.device

    # convert to within-head relative indices
    index = o_index.view(num_heads, head_channels)
    delta = torch.arange(
        0, num_channels, head_channels, device=device
    ).view(num_heads, 1)
    index = index - delta

    # first repeated heads
    index = index.view(kv_heads, num_repeats, head_channels)[:, 0, :]

    # add v_proj head offset
    num_kv_channels = num_channels // num_repeats
    delta = torch.arange(
        0, num_kv_channels, head_channels, device=device
    ).view(kv_heads, 1)
    index = index + delta

    return index.view(-1)