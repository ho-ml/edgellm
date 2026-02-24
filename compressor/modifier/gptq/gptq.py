import copy
import torch
import torch.nn as nn

from typing import *
from loguru import logger
from compressor.config import QuantArgs
from compressor.observers import Observer
from compressor.utils import fake_quantize
from compressor.modifier.weight.progressive import calculate_progressive_scales

__all__ = ["quantize_weight"]

def gptq(
    name: str,
    W: torch.Tensor,
    args: QuantArgs,
    H: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    block_size: int,
    perc_damp: float
):
    """
    Core GPTQ algorithms with pre-computed scale and zero
    """
    strategy = args.strategy
    num_rows = W.shape[0]
    num_cols = W.shape[1]

    # mask dead hessians
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    # activation ordering
    perm = torch.argsort(torch.diag(H), descending=True)
    W = W[:, perm]
    H = H[perm][:, perm]
    inv_perm = torch.argsort(perm)

    # compute inverse Hessian in-place
    try:
        damp = perc_damp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

    except torch._C._LinAlgError:
        logger.warning(f"[{name}] Cholesky failed! Falling back to RTN.")
        Hinv = H = torch.eye(num_cols, dtype=H.dtype, device=H.device)

    # initialize loss
    losses = torch.zeros(num_rows, device=W.device)
    for i1 in range(0, num_cols, block_size):
        i2 = min(i1 + block_size, num_cols)
        count = i2 - i1

        # initialize current block
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            # get current column
            w = W1[:, i]
            d = Hinv1[i, i]
            q = w.clone()

            # quantize column
            if strategy == "tensor":
                q = fake_quantize(q, scale, zero, args)
            elif strategy == "channel":
                q = fake_quantize(q, scale[:, 0], zero[:, 0], args)
            elif strategy == "group":
                group_idx = perm[i1 + i] // args.group_shapes[-1][-1]

                # channel-wise for this group
                altered_args = copy.copy(args)
                altered_args.strategy = "channel"
                q = fake_quantize(
                    q, scale[:, group_idx], zero[:, group_idx], altered_args
                )
            else:
                raise ValueError(f"Not supported strategy: {strategy}")

            # save quantized result of current column
            Q1[:, i] = q
            losses1[:, i] = (w - q) ** 2 / d ** 2

            # calculate error
            err1 = (w - q) / d
            w1_err = err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))

            # compensate quantized error of current columns
            W1[:, i:] -= w1_err
            Err1[:, i] = err1

        # save quantized result of current block
        W[:, i1:i2] = Q1
        losses += torch.sum(losses1, 1) / 2

        # compensate remained block error
        w_err = Err1.matmul(Hinv[i1:i2, i2:])
        W[:, i2:] -= w_err

    # restore original column order
    W = W[:, inv_perm]
    loss = torch.sum(losses).item()

    return loss, W, scale, zero    

def quantize_weight(
    name: str,
    module: nn.Module,
    args: QuantArgs,
    hessians_dict: dict[str, torch.Tensor],
    block_size: int = 128,
    perc_damp: float = 0.01
):
    """
    Quantize a module weight according to the GPTQ algorithm
    
    - pre-computes channel and group scales for progrssive group quantization
    - run gptq algorithms on the normalized weights
    """
    W = module.weight.clone()
    ori_shape, ori_dtype = module.weight.shape, module.weight.dtype
    H = hessians_dict[name]
    del hessians_dict[name]

    # cast dtype
    W = W.to(dtype=torch.float32)

    if args.is_progressive:
        # get progressive scales & float-normalized weights
        W, scale_0, scale_1, zero_1 = calculate_progressive_scales(W, args)

        # run gptq algorithm on float-normalized weights
        loss, W, _, _ = gptq(
            name, W, args, H, scale_1, zero_1, block_size, perc_damp
        )

        # normalize loss from W_norm space back to original weight space
        loss = loss * scale_0.pow(2).mean().item()        

        return (
            loss,
            W.reshape(ori_shape).to(ori_dtype),
            scale_0.to(ori_dtype),
            scale_1.to(ori_dtype),
            zero_1.to(ori_dtype)
        )

    # initialize scale & zero
    observer_cls = Observer.get(args.observer)
    observer = observer_cls(args=args)
    scale, zero = observer(W, target="weight")    

    # run gptq algorightm
    loss, W, scale, zero = gptq(
        name, W, args, H, scale, zero, block_size, perc_damp
    )

    return (
        loss,
        W.reshape(ori_shape).to(ori_dtype),
        scale.to(ori_dtype),
        zero.to(ori_dtype)
    )