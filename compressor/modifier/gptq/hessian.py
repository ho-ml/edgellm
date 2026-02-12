import math
import torch
import contextlib
import torch.nn as nn

from compressor.utils.device import get_execution_device

__all__ = ["init_hessian", "accumulate_hessian", "onload_hessian"]

def init_hessian(module: nn.Module):
    """
    Make empty hessian matrix
    """
    w = module.weight
    in_channels = w.shape[1]

    return torch.zeros(
        (in_channels, in_channels), device="cpu", dtype=torch.float32
    )

def accumulate_hessian(
    inp: torch.Tensor, 
    H: torch.Tensor | None, 
    num_samples: int
):
    """
    Accumulate hessian matrix using input activations
    """
    # for batch processing
    inp = inp.to(device=H.device)
    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    num_added = inp.shape[0]
    
    # reshape input tensor
    if len(inp.shape) == 3:
        inp = inp.reshape((-1, inp.shape[-1]))
    inp = inp.t()

    # update normalization
    H *= num_samples / (num_samples + num_added)
    num_samples += num_added

    # update matrix
    inp = inp.to(dtype=torch.float32)
    inp = math.sqrt(2 / num_samples) * inp
    H += inp.matmul(inp.t())

    return H, num_samples

@contextlib.contextmanager
def onload_hessian(name: str, module: nn.Module, hessian_dict: dict[str, torch.Tensor]):
    """
    Upload hessian to execution device and return back to cpu
    """
    device = get_execution_device(module)
    hessian_dict[name] = hessian_dict[name].to(device=device)

    yield

    if name in hessian_dict:
        hessian_dict[name] = hessian_dict[name].to(device="cpu")
        