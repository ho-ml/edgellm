import copy
import torch
import torch.nn as nn

from typing import *
from loguru import logger
from compressor.config import QuantArgs
from compressor.observers import Observer
from compressor.utils import fake_quantize

def gptq_quantize(
    name: str,
    module: nn.Module,
    quant_args: QuantArgs,
    hessians_dict: dict[str, torch.Tensor],
    block_size: int = 128,
    perc_damp: float = 0.01
):
    """
    Quantize a module weight according to the GPTQ algorithm
    """
    strategy = quant_args.strategy

    # current weights and hessian matrix
    W = module.weight.clone()
    ori_shape, ori_dtype = module.weight.shape, module.weight.dtype
    H = hessians_dict[name]
    del hessians_dict[name]

    # get shapes
    W = W.to(dtype=torch.float32)
    num_rows = W.shape[0]
    num_cols = W.shape[1]

    # get quantization params
    observer_cls = Observer.get(quant_args.observer)
    observer = observer_cls(args=quant_args)
    scale, zero = observer(W, target="weight")

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
        # add damping
        damp = perc_damp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp

        # Cholesky decomposition
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

    # fallback to RTN
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
                q = fake_quantize(
                    q, scale, zero, quant_args
                )
            elif strategy == "channel":
                q = fake_quantize(
                    q, scale[:, 0], zero[:, 0], quant_args
                )
            elif strategy == "group":
                group_idx = perm[i1 + i] // quant_args.group_size

                # channel-wise for this group
                altered_args = copy.copy(quant_args)
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

    # return back quantized weight to original shape & dtype
    W = W.reshape(ori_shape).to(dtype=ori_dtype)

    # calculate loss
    loss = torch.sum(losses).item()

    return (
        loss,
        W,
        scale.to(dtype=ori_dtype),
        zero.to(dtype=ori_dtype),
    )
