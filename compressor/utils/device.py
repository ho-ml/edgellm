import torch
import torch.nn as nn

__all__ = ["get_best_device", "get_execution_device", "move_to_device"]

def get_best_device():
    """
    Return available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_execution_device(module: nn.Module):
    """
    Return device of module (checks parameters first, then buffers)
    """
    return next(module.parameters()).device

def move_to_device(v, device: torch.device | str):
    """
    Move v to device
    """
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, tuple):
        return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in v)
    return v