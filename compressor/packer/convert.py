import torch

from typing import *
from compressor.config.quant import QuantArgs
from compressor.utils import quantize, pack_int4

__all__ = ["convert_linear"]
    
def linear_w4a16(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    args: QuantArgs,
    dtype: torch.dtype = torch.float16,
):
    """
    W4A16 linear

    - quantize weight
    - pack
    """
    # quantize weight
    q = quantize(
        weight.float(),
        scale.float(),
        zero.float(),
        args,
        dtype=torch.int8
    )

    # pack
    qweight =  pack_int4(q)

    return {
        "qweight": qweight,
        "scales": scale.to(dtype),
        "zeros": zero.to(dtype)
    }

def linear_w4a8(
    weight: torch.Tensor,
    scale_0: torch.Tensor,
    scale_1: torch.Tensor,
    zero: torch.Tensor,
    args: QuantArgs,
    dtype: torch.dtype = torch.float16,
):
    """
    W4A8 progressive linear

    - normalize weight by dividing scale_0
    - quantize weight
    - pack
    """
    level0_col = args.group_shapes[0][-1]

    # if level0 is per-channel
    if level0_col == -1:
        w_norm = weight.float() / scale_0.float()

    # if level0 is per-group
    else:
        oc, ic = weight.shape
        ng = ic // level0_col

        w_norm = weight.float().view(oc, ng, level0_col) / scale_0.float().unsqueeze(-1)
        w_norm = w_norm.view(oc, ic)

    # quantize
    q = quantize(
        w_norm.float(),
        scale_1.float(),
        zero.float(),
        args,
        dtype=torch.int8
    )

    # pack
    qweight =  pack_int4(q)

    return {
        "qweight": qweight,
        "scales_0": scale_0.to(dtype),
        "scales_1": scale_1.to(dtype),
        "zeros": zero.to(dtype)
    }

def linear_w8a8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    args: QuantArgs,
    dtype: torch.dtype = torch.float16,
):
    """
    W8A8 linear

    - quantize weight
    """
    # quantize
    q = quantize(
        weight.float(),
        scale.float(),
        zero.float(),
        args,
        dtype=torch.int8
    )

    return {
        "qweight": q,
        "scales": scale.to(dtype),
        "zeros": zero.to(dtype)
    }

def convert_linear(
    weight: torch.Tensor,
    qparams: dict[str, torch.Tensor],
    qparams_key: str,
    format_type: str,
    args: QuantArgs,
    dtype: torch.dtype = torch.float16,
):
    """
    Convert a single linear module based on format type
    """
    if format_type == "w4a16":
        scale = qparams[f"{qparams_key}.scale"]
        zero = qparams[f"{qparams_key}.zero"]

        return linear_w4a16(weight, scale, zero, args, dtype=dtype)

    elif format_type == "w4a8":
        scale_0 = qparams[f"{qparams_key}.scale_0"]
        scale_1 = qparams[f"{qparams_key}.scale_1"]
        zero = qparams[f"{qparams_key}.zero"]

        return linear_w4a8(weight, scale_0, scale_1, zero, args, dtype=dtype)

    elif format_type == "w8a8":
        scale = qparams[f"{qparams_key}.scale"]
        zero = qparams[f"{qparams_key}.zero"]

        return linear_w8a8(weight, scale, zero, args, dtype=dtype)

    else:
        raise ValueError(f"Unknown format: {format_type}")