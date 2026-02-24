import torch

from typing import *
from functools import lru_cache
from compressor.config.quant import QuantArgs
from compressor.utils import calculate_qparams

__all__ = ["calculate_progressive_scales", "restore_weight"]

def calculate_progressive_scales(x: torch.Tensor, args: QuantArgs):
    """
    Calculate multi-level scales for progressive quantization

    - compute the protective range for preventing overflow
    - compute level 0 scale (per-channel or per-group based on group_shapes[0])
    - compute level 1 scale & zero (per-group)
    """
    pmax = float(_get_protective_range(
        args.intermediate_bits, args.bits, args.symmetric, args.zero_domain
    ))

    level0_col = args.group_shapes[0][-1]
    # per-channel
    if level0_col == -1:
        scale_0 = x.abs().amax(dim=-1, keepdim=True)
        scale_0 = torch.clamp(scale_0 / pmax, min=torch.finfo(x.dtype).eps)
        x.div_(scale_0).clamp_(-pmax, pmax)
    
    # per-group at level 0
    else:
        x = x.unflatten(-1, (-1, level0_col))
        scale_0 = x.abs().amax(dim=-1, keepdim=True)
        scale_0 = torch.clamp(scale_0 / pmax, min=torch.finfo(x.dtype).eps)
        x = x.div_(scale_0).clamp_(-pmax, pmax).flatten(-2)

    # reshape for level 1
    group_size = args.group_shapes[-1][-1]
    x_reshaped = x.unflatten(-1, (-1, group_size))

    # compute level 1 scale & zero
    scale_1, zero_1 = calculate_qparams(
        min_vals=x_reshaped.amin(dim=-1),
        max_vals=x_reshaped.amax(dim=-1),
        args=args
    )

    return x, scale_0, scale_1, zero_1


def restore_weight(
    qweight: torch.Tensor, scale: torch.Tensor, col: int
):
    """
    Restore weight from normalized space using level0 scale
    """
    qweight = qweight.to(torch.float32)
    if col == -1:
        return qweight * scale
    else:
        return (qweight.unflatten(-1, (-1, col)) * scale).flatten(-2)

@lru_cache(maxsize=None)
def _get_protective_range(
    intermediate_bits: int, bits: int, symmetric: bool, zero_domain: Optional[str]
):
    """
    Compute the protective bound for the intermediate dtype
    """
    # level 0 ranges
    max0 = 2 ** (intermediate_bits - 1) - 1
    min0 = -(2 ** (intermediate_bits - 1))

    # if level1 is symmetric, no overflow
    if symmetric: return max0

    # level 1 range
    max1 = 2 ** (bits - 1) - 1
    min1 = -(2 ** (bits - 1))
    qrange = max1 - min1

    # build set of (vmin, vmax)
    vranges = set()
    for vmax in range(0, max0 + 1):
        for vmin in range(min0, vmax + 1):
            # compute scales
            s = round((vmax - vmin) / qrange)
            s = min(max(1, s), max0)

            # pre-scale: need to clamp z to [min1, max1]
            if zero_domain == "PreScale":
                # compute zero point
                z = round(min1 - vmin / s)
                z = max(min(z, max1), min1)

                # fake quantized max
                mx = round(vmax / s + z)
                mx = max(min(mx, max1), min1)
                mx = (mx - z) * s

                # fake quantized min
                mn = round(vmin / s + z)
                mn = max(min(mn, max1), min1)
                mn = (mn - z) * s

            # post scale: need to clamp to [min0, max0]
            elif zero_domain == "PostScale":
                # compute scaled zero point
                z = round(min1 * s - vmin)
                z = max(min(z, max0), min0)

                # fake quantized max
                mx = round((vmax + z) / s)
                mx = max(min(mx, max1), min1)
                mx = mx * s - z

                # fake quantized min
                mn = round((vmin + z) / s)
                mn = max(min(mn, max1), min1)
                mn = mn * s - z

            else:
                raise ValueError(f"Unsupported zero domain: {zero_domain}")

            # if satisfied, append to candidates
            if min0 <= mx <= max0 and min0 <= mn <= max0:
                vranges.add((vmin, vmax))

    # find largest pmax
    for pmax in range(max0, 0, -1):
        is_valid = True

        for vmax in range(0, pmax + 1):
            for vmin in range(-pmax, vmax + 1):
                # check if (vmin, vmax) in candidates
                if (vmin, vmax) not in vranges:
                    is_valid = False
                    break

            # not in candidates
            if not is_valid: break

        # in candidates
        if is_valid: return pmax

    return 1
