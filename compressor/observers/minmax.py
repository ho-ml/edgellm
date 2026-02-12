import torch

from compressor.observers.base import Observer
from compressor.config.quant import QuantArgs

__all__ = ["MinMaxObserver", "MemorylessMinMaxObserver"]

@Observer.register("minmax")
class MinMaxObserver(Observer):
    """
    Static minmax observer
    """
    def __init__(self, args: QuantArgs, **kwargs):
        super().__init__(args, **kwargs)
        
        self.past_min_vals = None
        self.past_max_vals = None

    def get_min_max(self, x: torch.Tensor):
        """
        Accumulates statistics across batches
        """
        min_vals, max_vals = _get_min_max(x)

        if self.past_min_vals is not None:
            min_vals = torch.min(min_vals, self.past_min_vals)
            max_vals = torch.max(max_vals, self.past_max_vals)

        self.past_min_vals = min_vals
        self.past_max_vals = max_vals

        return min_vals, max_vals

@Observer.register("memoryless-minmax")
class MemorylessMinMaxObserver(Observer):
    """
    Memoryless MinMax Observer
    """
    def get_min_max(self, x: torch.Tensor):
        """
        Computes per-batch statistics
        """
        return _get_min_max(x)

def _get_min_max(x: torch.Tensor):
    min_vals = torch.amin(x, dim=(0, -1))
    max_vals = torch.amax(x, dim=(0, -1))

    return min_vals, max_vals