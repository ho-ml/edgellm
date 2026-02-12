import functools
import torch
import torch.nn as nn

from loguru import logger

__all__ = ["patch_norm"]


def gemma_rms_norm_forward(self, x: torch.Tensor):
    """
    Forward for patched Gemma RMSNorm
    """
    output = self._norm(x.float())
    output = output * self.weight.float()

    return output.type_as(x)


def patch_norm(model: nn.Module):
    """
    Patch Gemma RMSNorm modules to use standard weight convention
    """
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
    except ImportError:
        GemmaRMSNorm = None

    try:
        from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
    except ImportError:
        Gemma2RMSNorm = None

    # get norm class
    norm_classes = tuple(c for c in (GemmaRMSNorm, Gemma2RMSNorm) if c is not None)
    if not norm_classes:
        return model

    for module_name, module in model.named_modules():
        if not isinstance(module, norm_classes):
            continue

        classname = type(module).__name__
        forward_name = ""
        if hasattr(module, "_edgellm_orig_forward"):
            logger.info(f"- {module_name} has already been patched")
        elif isinstance(module.forward, functools.partial):
            assert hasattr(module, "_old_forward")
            assert module._old_forward is module.forward.__wrapped__
            forward_name = "_old_forward"
        else:
            forward_name = "forward"

        if forward_name:
            module.weight.data.add_(1.0)
            module._edgellm_orig_forward = getattr(module, forward_name)
            setattr(module, forward_name, functools.partial(gemma_rms_norm_forward, module))

    return model
