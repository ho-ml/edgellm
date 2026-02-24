import math
import torch

from typing import *
from compressor.config.quant import QuantArgs

__all__ = [
    "calculate_range",
    "calculate_qparams",
    "fake_quantize",
    "quantize",
    "dequantize",
]

def calculate_range(bits: int, device: torch.device):
    """
    Calculate ranges according to bits
    """
    bit_range = 2 ** bits
    qmax = torch.tensor(bit_range / 2 - 1, device=device)
    qmin = torch.tensor(-bit_range / 2, device=device)

    return qmax, qmin

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
    qmax, qmin = calculate_range(args.bits, device)
    qrange = qmax - qmin

    # genreate scale & zero point
    if args.symmetric:
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        scales = max_val_pos / (float(qrange) / 2)
        zeros = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)
    else:
        scales = (max_vals - min_vals) / float(qrange)
        if args.zero_domain == "PostScale":
            zeros = qmin * scales - min_vals
            if args.intermediate_bits is not None:
                imax = 2 ** (args.intermediate_bits - 1) - 1
                imin = -(2 ** (args.intermediate_bits - 1))
                zeros = torch.clamp(zeros, imin, imax)
        else:
            zeros = qmin - (min_vals / scales)
            zeros = torch.clamp(zeros, qmin, qmax)

    # prevent div by 0
    eps = torch.finfo(scales.dtype).eps
    scales = torch.clamp(scales, min=eps)

    # round the zero points
    zeros = torch.round(zeros)

    # reshape dimensions
    if scales.ndim == 0:
        scales = scales.reshape(1)
        zeros = zeros.reshape(1)

    return scales, zeros

@torch.no_grad()
def quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    args: QuantArgs,
    dtype: Optional[torch.dtype] = None
):
    """
    Quantize the input tensor x
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero=zero,
        args=args,
        dtype=dtype,
        do_quantize=True,
        do_dequantize=False
    )

@torch.no_grad()
def dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero: Optional[torch.Tensor] = None,
    args: Optional[QuantArgs] = None,
    dtype: Optional[torch.dtype] = None
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
                args = QuantArgs(strategy="group", group_shapes=[[1, group_size]])
        else:
            raise ValueError(
                f"Could not infer a strategy from scale with {scale.ndim}"
            )

    if dtype is None:
        dtype = scale.dtype

    return _process_quantization(
        x=x_q,
        scale=scale,
        zero=zero,
        args=args,
        do_quantize=False,
        do_dequantize=True,
        dtype=dtype
    )

@torch.no_grad()
def fake_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    args: QuantArgs
):
    """
    Fake quantize the input tensor x by quantizing then dequantizing
    """
    return _process_quantization(
        x=x,
        scale=scale,
        zero=zero,
        args=args,
        do_quantize=True,
        do_dequantize=True
    )

@torch.no_grad()
def _process_quantization(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    args: QuantArgs,
    dtype: Optional[torch.dtype] = None,
    do_quantize: bool = True,
    do_dequantize: bool = True
):
    # calculate range
    qmax, qmin = calculate_range(args.bits, x.device)

    # group size
    group_size = args.group_shapes[-1][-1] if args.group_shapes is not None else 128

    # quantization
    if args.strategy == "group":
        output_dtype = dtype if dtype is not None else x.dtype
        output = torch.zeros_like(x).to(output_dtype)
        columns = output.shape[-1]

        # pad scale and zero point dims for slicing
        while scale.ndim < 2:
            scale = scale.unsqueeze(1)
            zero = zero.unsqueeze(1) if zero is not None else None

        # sanity check
        if columns >= group_size:
            if columns % group_size != 0:
                raise ValueError(f"column shape must be divisble group_size {group_size}")
        
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
                zero=zero.unsqueeze(-1) if zero is not None else None,
                dtype=dtype,
                qmin=qmin,
                qmax=qmax,
                zero_domain=args.zero_domain,
            )

        if do_dequantize:
            input = output if do_quantize else x
            output = _dequantize(
                x_q=input,
                scale=scale.unsqueeze(-1),
                zero=zero.unsqueeze(-1) if zero is not None else None,
                zero_domain=args.zero_domain,
            )

        output = output.flatten(start_dim=-2)
        output = output.to(output_dtype)

    # covers tensor, channel, token, and attn_head strategies
    else:
        if do_quantize:
            output = _quantize(
                x=x,
                scale=scale,
                zero=zero,
                qmin=qmin,
                qmax=qmax,
                dtype=dtype,
                zero_domain=args.zero_domain,
            )

        if do_dequantize:
            output = _dequantize(
                output if do_quantize else x,
                scale=scale,
                zero=zero,
                zero_domain=args.zero_domain,
            )

    return output


@torch.no_grad()
def _quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    qmin: torch.Tensor,
    qmax: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
    zero_domain: Optional[str] = None,
):
    if zero_domain == "PostScale":
        scaled = (x + zero.to(x.dtype)) / scale
    else:
        scaled = x / scale
        if zero is not None:
            scaled += zero.to(x.dtype)

    # clamp and round
    q = torch.clamp(torch.round(scaled), qmin, qmax)

    if dtype is not None:
        q = q.to(dtype)

    return q

@torch.no_grad()
def _dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor = None,
    dtype: Optional[torch.dtype] = None,
    zero_domain: Optional[str] = None,
):
    dq = x_q.to(scale.dtype)

    if zero_domain == "PostScale":
        dq = dq * scale
        if zero is not None:
            dq = dq - zero.to(scale.dtype)
    else:
        if zero is not None:
            dq = dq - zero.to(scale.dtype)
        dq = dq * scale

    if dtype is not None:
        dq = dq.to(dtype)

    return dq
