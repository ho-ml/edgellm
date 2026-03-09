import torch

from typing import *
from abc import abstractmethod
from compressor.utils import RegistryMixin
from compressor.config.quant import QuantArgs, QuantConfig

__all__ = ["FormatHandler", "infer_format"]

class FormatHandler(RegistryMixin):
    """
    Base class for handling various data types
    """
    def __init__(self, args: QuantArgs):
        self.args = args

    @abstractmethod
    def convert_linear(
        self, 
        weight: torch.Tensor,
        qparams: Dict[str, torch.Tensor],
        qparams_key: str,
        dtype: torch.dtype
    ):
        """
        Convert linear module to compressed module
        """
        ...

    @abstractmethod
    def pack_weight(
        self,
        weight: torch.Tensor,
        nrows: int,
        ncols: int
    ):
        """
        Pack weight according to formats
        """
        ...

    @abstractmethod
    def pack_params(
        self,
        scale: torch.Tensor,
        zero: torch.Tensor,
        nrows: int,
        ncols: int,
        group_size: int
    ):
        """
        Pack quantized params according to formats
        """
        ...
        
    def _get_group_size(self, size: int):
        """
        Extract group size from args 
        """
        # per-group
        if self.args.group_shapes is not None and len(self.args.group_shapes) > 0:
            gs = self.args.group_shapes[-1][-1]
            if gs == -1: return size
            else: return gs
        
        # per-channel
        if self.args.strategy == "channel":
            return size

        raise ValueError()


def infer_format(quant_config: QuantConfig):
    """
    Infer format scheme from quantization config
    """
    wbits = quant_config.weight.bits
    has_input = quant_config.input is not None

    if wbits == 4:
        if has_input:
            return "w4a8"
        else:
            return "w4a16"

    elif wbits == 8:
        if has_input:
            return "w8a8"

    raise ValueError("Unsupported format")

