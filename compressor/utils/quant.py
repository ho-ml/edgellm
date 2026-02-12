import math
import torch

from typing import *
from compressor.config.quant import QuantArgs

__all__ = [
    "calculate_qparams", 
    "fake_quantize",
    "quantize",
    "dequantize"
]

def calculate_qparams(
    min_vals: torch.Tensor, 
    max_vals: torch.Tensor,
    args: QuantArgs
):
    """
    Calculate scale(s) and zero point(s)
    """
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

    device = min_vals.device

    # calculate range
    bit_range = 2 ** args.bits
    qmax = torch.tensor(bit_range / 2 - 1, device=device)
    qmin = torch.tensor(-bit_range / 2, device=device)
    qrange = qmax - qmin

    # genreate scale & zero point
    if args.symmetric:
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        scales = max_val_pos / (float(qrange) / 2)
        zero_points = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)
    else:
        scales = (max_vals - min_vals) / float(qrange)
        zero_points = qmin - (min_vals / scales)
        zero_points = torch.clamp(zero_points, qmin, qmax)

    # prevent div by 0
    eps = torch.finfo(scales.dtype).eps
    scales = torch.clamp(scales, min=eps)

    # round the zero points
    zero_points = torch.round(zero_points)

    # reshape dimensions
    if scales.ndim == 0:
        scales = scales.reshape(1)
        zero_points = zero_points.reshape(1)

    return scales, zero_points

@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantArgs,
    dtype: Optional[torch.dtype] = None,
    g_idx: Optional[torch.Tensor] = None
):
    """
    Quantize the input tensor x
    """

    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        dtype=dtype,
        do_quantize=True,
        do_dequantize=False,
        g_idx=g_idx
    )

@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    args: Optional[QuantArgs] = None,
    dtype: Optional[torch.dtype] = None,
    g_idx: Optional[torch.Tensor] = None
):
    """
    Dequantize a quantized input tensor x_q
    """
    if args is None:
        if scale.ndim == 0 or scale.ndim == 1:
            args = QuantArgs(strategy="tensor")
        elif scale.ndim == 2:
            if scale.shape[1] == 1:
                args = QuantArgs(strategy="channel")
            elif (scale.shape[0] == 1) or (scale.shape[0] == x_q.shape[0]):
                group_size = int(x_q.shape[1] / scale.shape[1])
                args = QuantArgs(strategy="group", group_size=group_size)
        else:
            raise ValueError(
                f"Could not infer a strategy from scale with {scale.ndim}"
            )

    if dtype is None:
        dtype = scale.dtype

    return _process_quantization(
        x=x_q,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=False,
        do_dequantize=True,
        dtype=dtype,
        g_idx=g_idx
    )

@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantArgs,
    g_idx: Optional[torch.Tensor] = None
):
    """
    Fake quantize the input tensor x by quantizing then dequantizing
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero_point=zero_point,
        args=args,
        do_quantize=True,
        do_dequantize=True,
        g_idx=g_idx
    )

@torch.no_grad()
def _process_quantization(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    args: QuantArgs,
    g_idx: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    do_quantize: bool = True,
    do_dequantize: bool = True
):
    # calculate range
    bit_range = 2 ** args.bits
    qmax = torch.tensor(bit_range / 2 - 1, device=x.device)
    qmin = torch.tensor(-bit_range / 2, device=x.device)

    # group size
    group_size = args.group_size
    
    # quantization
    if args.strategy == "group":
        output_dtype = dtype if dtype is not None else x.dtype
        output = torch.zeros_like(x).to(output_dtype)
        columns = output.shape[-1]

        # pad scale and zero point dims for slicing
        while scale.ndim < 2:
            scale = scale.unsqueeze(1)
            zero_point = zero_point.unsqueeze(1) if zero_point is not None else None

        # sanity check
        if columns >= group_size:
            if columns % group_size != 0:
                raise ValueError(f"column shape must be divisble group_size {group_size}")

        # support column-order quantization as well as other orderings
        is_column_order = g_idx is None or -1 in g_idx
        if is_column_order:
            num_groups = int(math.ceil(columns / group_size))
            group_sizes = torch.full((num_groups,), group_size, dtype=torch.int)
        else:
            group_indices, group_sizes = torch.unique(g_idx, return_counts=True)
            group_sizes = group_sizes[torch.argsort(group_indices)]

            perm = torch.argsort(g_idx)
            x = x.index_select(-1, perm)

        # maintain all dims except the last dim
        reshaped_dims = (
            math.ceil(x.shape[-1] / group_size),
            group_size,
        )
        x = x.unflatten(-1, reshaped_dims)

        if do_quantize:
            output = _quantize(
                x=x,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
                dtype=dtype,
                qmin=qmin,
                qmax=qmax,
            )

        if do_dequantize:
            input = output if do_quantize else x
            output = _dequantize(
                x_q=input,
                scale=scale.unsqueeze(-1),
                zero_point=zero_point.unsqueeze(-1) if zero_point is not None else None,
            )

        output = output.flatten(start_dim=-2)
        output = output.to(output_dtype)

        if not is_column_order:
            inv_perm = torch.argsort(perm)
            output = output.index_select(-1, inv_perm)

    # covers tensor, channel, token, and attn_head strategies
    else:
        if do_quantize:
            output = _quantize(
                x=x,
                scale=scale,
                zero_point=zero_point,
                qmin=qmin,
                qmax=qmax,
                dtype=dtype
            )
        
        if do_dequantize:
            output = _dequantize(
                output if do_quantize else x,
                scale=scale,
                zero_point=zero_point
            )

    return output


@torch.no_grad()
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: torch.Tensor,
    qmax: torch.Tensor,
    dtype: Optional[torch.dtype] = None
):
    scaled = x / scale

    if zero_point is not None:
        scaled += zero_point.to(x.dtype)

    # clamp and round
    q = torch.clamp(torch.round(scaled), qmin, qmax)

    if dtype is not None:
        q = q.to(dtype)

    return q

@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor = None,
    dtype: Optional[torch.dtype] = None
):
    dq = x_q.to(scale.dtype)

    if zero_point is not None:
        dq = dq - zero_point.to(scale.dtype)
    dq = dq * scale

    if dtype is not None:
        dq = dq.to(dtype)

    return dq