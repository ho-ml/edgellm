import torch
import torch.nn as nn

from typing import *

__all__ = [
    "rotate_in",
    "rotate_out",
    "fuse_ln_fcs",
    "convert_ln_to_rms"
]

def rotate_in(weight: nn.Parameter, rotation: torch.Tensor):
    """
    Rotates the input channels of a weight matrix
    to compensate for the rotation of input activations

    Y = X · W.T → Y = X · (W · H).T = X · H.T · W.T
    """
    shape, dtype = weight.shape, weight.dtype

    w = weight.data.view(-1, rotation.shape[0]).to(dtype=torch.float64)
    h = rotation.to(weight.device)

    weight.data = torch.matmul(w, h).to(dtype=dtype).view(shape)

def rotate_out(weight: nn.Parameter, rotation: torch.Tensor, bias: nn.Parameter | None = None):
    """
    Rotates the output channels of a weight matrix 
    to apply rotation to the output activations

    Y = X · W.T → Y = X · (H.T · W).T = X · W.T · H
    """
    shape, dtype = weight.shape, weight.dtype
    out_channels, head_channels = shape[0], rotation.shape[0]
    num_heads = out_channels // head_channels

    w = weight.data.view(num_heads, head_channels, -1).to(dtype=torch.float64)
    h = rotation.T.to(weight.device)

    # weight
    weight.data = torch.matmul(h, w).to(dtype=dtype).view(shape)

    # bias
    if bias is not None:
        b = bias.data.view(num_heads, head_channels, -1).to(dtype=torch.float64)
        bias.data = torch.matmul(h, b).to(dtype=dtype).view(-1)

def fuse_ln_fcs(ln: nn.Module, fcs: List[nn.Linear]):
    """
    Fuse the norm weight (gamma) into the adjacent linear modules
    """
    # normalize norm weight & bias
    gamma = ln.weight.data.to(dtype=torch.float64)
    ln.weight.data = torch.ones_like(ln.weight.data)
    if hasattr(ln, "bias") and ln.bias is not None:
        beta = ln.bias.data.to(dtype=torch.float64)
        ln.bias = None
    else:
        beta = None

    # fuse adjacent linear weights
    for fc in fcs:
        dtype = fc.weight.dtype
        device = fc.weight.device

        # weight
        w = fc.weight.data.to(dtype=torch.float64)
        gamma = gamma.to(device)
        fc.weight.data = (w * gamma).to(dtype=dtype)
        
        # bias
        if beta is not None:
            beta = beta.to(device)
            if fc.bias is None:
                fc.bias = nn.Parameter(
                    torch.zeros(fc.out_features, dtype=dtype, device=device)
                )
            b = fc.bias.to(dtype=torch.float64)
            fc.bias.data = (b + torch.matmul(w, beta)).to(dtype=dtype)

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps    
    
    def forward(self, x: torch.Tensor):
        dtype = x.dtype

        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(var + self.eps)

        return self.weight * out.to(dtype)
        
def convert_ln_to_rms(
    parent: nn.Module,
    norm_rname: str,
    modules: list[nn.Module] | nn.Module,
    out_dims: List[int] | int = 0,
):
    """
    Convert LayerNorm to RMSNorm by moving mean substraction to prev modules
    
    Y' = xW' + b'
    W' = W - mean(W, dim=output), b' = b - mean(b)
    """
    # get norm modules
    norm_parts = norm_rname.split(".")
    norm_parent = parent
    for part in norm_parts[:-1]:
        norm_parent = getattr(norm_parent, part)
    norm_name = norm_parts[-1]
    norm = getattr(norm_parent, norm_name)

    # sanity check
    if not isinstance(norm, nn.LayerNorm):
        return None
    if hasattr(norm, "bias") and norm.bias is not None:
        raise ValueError("LayerNorm bias should be None.")

    # unify the type
    if isinstance(modules, nn.Module):
        modules = [modules]
    if isinstance(out_dims, int):
        out_dims = [out_dims] * len(modules)

    # move mean substraction to prev modules
    for module, dim in zip(modules, out_dims, strict=True):
        dtype = module.weight.dtype

        # weight
        w = module.weight.data.to(dtype=torch.float64)
        module.weight.data = w.sub_(w.mean(dim=dim, keepdim=True)).to(dtype=dtype)

        # bias
        if hasattr(module, "bias") and module.bias is not None:
            b = module.bias.data.to(dtype=torch.float64)
            module.bias.data = b.sub_(b.mean()).to(dtype=dtype)

    # initialize RMSNorm layer
    hidden_size = norm.normalized_shape[0]
    eps = norm.eps
    rms = RMSNorm(hidden_size=hidden_size, eps=eps)

    # convert to RMSNorm
    rms.weight.data = norm.weight.data
    setattr(norm_parent, norm_name, rms)

    return rms

    
