import torch

from dataclasses import dataclass
from compressor.config.quant import QuantArgs
from compressor.observers import Observer
from compressor.utils import PersistentHook, fake_quantize

__all__ = ["DynamicInputQuantHook", "DynamicOutputQuantHook"]


@dataclass
class DynamicInputQuantHook(PersistentHook):
    """
    Forward pre hook for dynamic input activation quantization
    """
    args: QuantArgs

    def process(self, x: torch.Tensor):
        observer_cls = Observer.get(self.args.observer)
        observer = observer_cls(self.args).to(x.device)
        scale, zero = observer(x, target="activation")
        scale, zero = _reshape_scale_zero(scale, zero, self.args.strategy)

        return fake_quantize(x, scale, zero, self.args)


@dataclass
class DynamicOutputQuantHook(PersistentHook):
    """
    Forward hook for dynamic output activation quantization
    """
    args: QuantArgs

    def process(self, x: torch.Tensor):
        observer_cls = Observer.get(self.args.observer)
        observer = observer_cls(self.args).to(x.device)
        scale, zero = observer(x, target="activation")
        scale, zero = _reshape_scale_zero(scale, zero, self.args.strategy)

        return fake_quantize(x, scale, zero, self.args)

def _reshape_scale_zero(scale, zero, strategy):
    """
    Reshape scale/zero for activation broadcasting
    """
    if strategy == "token":
        # (seq,) -> (1, seq, 1)
        scale = scale.unsqueeze(0).unsqueeze(-1)
        zero = zero.unsqueeze(0).unsqueeze(-1)
    
    elif strategy == "group":
        # (seq, num_groups) → (1, seq, num_groups)
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
            zero = zero.unsqueeze(0)

        # (num_groups,) → (1, 1, num_groups)
        else:
            scale = scale.unsqueeze(0).unsqueeze(0)
            zero = zero.unsqueeze(0).unsqueeze(0)

    return scale, zero