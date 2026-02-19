import torch
import torch.nn as nn

from typing import *
from abc import abstractmethod
from compressor.utils import RegistryMixin, calculate_qparams
from compressor.config.quant import QuantArgs

__all__ = ["Observer"]

class Observer(nn.Module, RegistryMixin):
    """
    Base class for observers which compute quantization parameters.
    """
    def __init__(self, args: QuantArgs, **kwargs):
        """
        Initialize observer with quantization args
        """
        super().__init__()
        self.args = args
        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(kwargs)

    @abstractmethod
    def get_min_max(self, x: torch.Tensor):
        """
        Calculate min & max values from observed value
        """
        pass
    
    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        target: str = "activation"
    ):
        """
        Calculate updated scales and zero points from observed value
        """
        x_flat = self._flatten_tensor(x, target)
        min_vals, max_vals = self.get_min_max(x_flat)

        # calculate scale & zeros
        scales, zero_points = calculate_qparams(
            min_vals=min_vals,
            max_vals=max_vals,
            args=self.args
        )

        return scales, zero_points
    
    def _flatten_tensor(
        self,
        x: torch.Tensor,
        target: str = "activation"
    ):
        """
        Flatten tensor according to strategy and target type
        """
        strategy = self.args.strategy

        # (out_feats, in_feats)
        if target == "weight":
            # (1, 1, out_feats * in_feats)
            if strategy == "tensor":
                return x.reshape((1, 1, -1))
            
            # (1, out_feats, 1, in_feats)
            elif strategy == "channel":
                return x.unsqueeze(-2).unsqueeze(0)
            
            # (1, out_feats, num_groups, group_size)
            elif strategy == "group":
                return x.unflatten(-1, (-1, self.args.group_size)).unsqueeze(0)
            
            else:
                raise ValueError(f"Unsupported strategy {strategy} in {target}")
        
        # (batch_size, seq_len, hidden_dim)
        elif target == "activation":
            # (batch_size * seq_len, 1, hidden_dim)
            if strategy == "tensor":
                return x.reshape(-1, 1, x.size(-1))
            
            # (batch_size, seq_len, hidden_dim)
            elif strategy == "token":
                return x
            
            # (batch_size * seq_len, num_groups, group_size)
            # elif strategy == "group":
            #     return x.flatten(0, 1).unflatten(-1, (-1, self.args.group_size))

            # (batch_size, seq_len, num_groups, group_size)
            elif strategy == "group":
                return x.unflatten(-1, (-1, self.args.group_size))

            else:
                raise ValueError(f"Unsupported strategy {strategy} in {target}")

        else:
            raise ValueError(f"Unknown target {target}")