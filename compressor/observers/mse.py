import copy
import torch

from typing import *
from compressor.observers.base import Observer
from compressor.config.quant import QuantArgs
from compressor.utils import calculate_qparams, fake_quantize

__all__ = ["MSEObserver"]

@Observer.register("mse")
class MSEObserver(Observer):
    """
    MSE observer
    """
    def __init__(self, args: QuantArgs, **kwargs):
        """
        Initialize MSE observer
        """
        super().__init__(args, **kwargs)
        observer_kwargs = self.args.observer_kwargs
        self.args.observer_kwargs.update(observer_kwargs)

        self.maxshrink = observer_kwargs.get("maxshrink", 0.20)
        self.patience = observer_kwargs.get("patience", 5)
        self.grid = observer_kwargs.get("grid", 100.0)
        self.norm = observer_kwargs.get("norm", 2.4)

    def get_min_max(self, x: torch.Tensor):
        """
        Finds optimal clipping via MSE minimization
        """
        return _search(
            x,
            self.args,
            self.maxshrink,
            self.patience,
            self.grid,
            self.norm
        )
    
def _search(
    x: torch.Tensor,
    args: QuantArgs,
    maxshrink: float,
    patience: float,
    grid: float,
    norm: float
):
    """
    Grid search to find per-channel min/max ranges
    """
    min_val = torch.amin(x, dim=(0, -1))
    max_val = torch.amax(x, dim=(0, -1))

    # initialize
    best_error = torch.full_like(min_val, torch.finfo(min_val.dtype).max)
    best_min_val = min_val.clone()
    best_max_val = max_val.clone()
    cnt = 0

    # strategy
    search_args = copy.deepcopy(args)
    search_args.strategy = "token"

    # grid search
    for i in range(int(maxshrink * grid)):
        p = 1 - i / grid
        smin = p * min_val
        smax = p * max_val

        # calculate scale & zero points
        candidate_scales, candidate_zero_points = calculate_qparams(
            min_vals=smin,
            max_vals=smax,
            args=args
        )

        # fake quantize
        q = fake_quantize(
            x,
            candidate_scales.unsqueeze(-1),
            candidate_zero_points.unsqueeze(-1),
            args
        ).to(x.dtype)

        # calculate error
        q -= x
        q.abs_()
        q.pow_(norm)
        err = torch.sum(q, dim=(0, -1))

        # update
        tmp = err < best_error
        if torch.any(tmp):
            best_error[tmp] = err[tmp]
            best_min_val[tmp] = smin[tmp]
            best_max_val[tmp] = smax[tmp]
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

    return best_min_val, best_max_val
        

