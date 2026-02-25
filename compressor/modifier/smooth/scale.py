import torch
import torch.nn as nn

__all__ = [
    "compute_scale", "scale_up", "scale_down",
]

def compute_scale(amax: torch.Tensor, wmax: torch.Tensor, alpha: float = 0.5, beta: float = 0.5):
    """
    Compute smooth scale factor

    s = amax ^ alpha / wmax ^ beta
    """
    amax = amax.clamp(min=1e-8)
    wmax = wmax.clamp(min=1e-8)

    scale = (amax ** alpha) / (wmax ** beta)
    return scale.clamp(min=1e-8)

def scale_up(param: nn.Parameter, scale: torch.Tensor, dim: int = 0):
    """
    Upscale parameter by scale factor

    param' = param * scale
    """
    shape = [1] * param.dim()
    shape[dim] = -1
    
    param.data = param.data * scale.view(shape).to(param.device, param.dtype)

def scale_down(param: nn.Parameter, scale: torch.Tensor, dim: int = 0):
    """
    Downscale parameter by scale factor

    param' = param / scale
    """
    shape = [1] * param.dim()
    shape[dim] = -1

    param.data = param.data / scale.view(shape).to(param.device, param.dtype)