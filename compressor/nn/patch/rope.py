import functools
import torch
import torch.nn as nn

from loguru import logger
from compressor.nn.patch.utils import copy_func

__all__ = ["RotaryEmbedding", "patch_rope"]

def rotate_half(x: torch.Tensor):
    """
    Rotates half the hidden dims of the input
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    return torch.cat((-x2, x1), dim=-1)

def update_rotary_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor | None,
    unsqueeze_dim: int = 1,
):
    """
    Update cos/sin tensors based on position_ids
    """
    assert unsqueeze_dim in (1, 2), f"unsqueeze_dim must be 1 or 2, got {unsqueeze_dim}"
    if position_ids is None:
        if cos.ndim == 2:
            cos = cos.unsqueeze(0)
        if sin.ndim == 2:
            sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    assert cos.ndim == 4, f"cos must have 4 dimensions, got {cos.ndim}"
    assert sin.ndim == 4, f"sin must have 4 dimensions, got {sin.ndim}"
    
    return cos, sin

class RotaryEmbedding(nn.Module):
    """
    Rotary embedding as nn.Module
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
    ):
        """
        Apply rotary embedding to states
        """
        states = (states * cos) + (rotate_half(states) * sin)
        if unsqueeze_dim == 1:
            batch_size, num_heads, seq_len, head_dim = states.shape
            states = states.transpose(1, 2)
        else:
            batch_size, seq_len, num_heads, head_dim = states.shape
        
        return states.view(batch_size, seq_len, num_heads * head_dim)


def apply_rotary_pos_emb(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.LongTensor = None,
    unsqueeze_dim: int = 1,
):
    """
    Apply Rotary Position Embedding to query and key tensors
    """
    assert unsqueeze_dim in (1, 2), f"unsqueeze_dim must be 1 or 2, got {unsqueeze_dim}"
    if unsqueeze_dim == 1:
        batch_size, _, seq_len, head_dim = q.shape
    else:
        batch_size, seq_len, _, head_dim = q.shape

    cos, sin = update_rotary_cos_sin(cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    
    q = self.q_rotary_emb(q, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)
    k = self.k_rotary_emb(k, cos=cos, sin=sin, unsqueeze_dim=unsqueeze_dim)

    q = q.view(batch_size, seq_len, -1, head_dim)
    k = k.view(batch_size, seq_len, -1, head_dim)
    
    if unsqueeze_dim == 1:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
    
    return q, k


def patch_rope(model: nn.Module):
    """
    Patch attention modules to use RotaryEmbedding nn.Modules
    """
    for module_name, module in model.named_modules():
        classname = type(module).__name__
        if not classname.lower().endswith("attention"):
            continue

        forward_name = ""
        if isinstance(module.forward, functools.partial):
            if hasattr(module, "_edgellm_orig_forward"):
                logger.info(f"- Attention in {module_name} has already been patched")
            else:
                assert hasattr(module, "_old_forward")
                assert module._old_forward is module.forward.__wrapped__
                if "apply_rotary_pos_emb" in module._old_forward.__func__.__globals__:
                    forward_name = "_old_forward"
        
        else:
            if hasattr(module, "_edgellm_orig_forward"):
                logger.info(f"- Attention in {module_name} has already been patched")
            elif "apply_rotary_pos_emb" in module.forward.__func__.__globals__:
                forward_name = "forward"

        if forward_name:
            module.q_rotary_emb = RotaryEmbedding()
            module.k_rotary_emb = RotaryEmbedding()
            
            module.apply_rotary_pos_emb = functools.partial(apply_rotary_pos_emb, module)
            module._edgellm_orig_forward = getattr(module, forward_name)

            orig_forward = module._edgellm_orig_forward.__func__
            new_globals = dict(orig_forward.__globals__)
            new_globals["apply_rotary_pos_emb"] = module.apply_rotary_pos_emb
            new_forward = copy_func(orig_forward, new_globals)
            setattr(module, forward_name, new_forward.__get__(module))

    return model
