import torch
import numpy as np

from typing import *
from compressor.utils import quantize
from compressor.config.quant import QuantArgs
from compressor.packer.format.base import FormatHandler

__all__ = ["W8A8Format"]

@FormatHandler.register("w8a8")
class W8A8Format(FormatHandler):
    """
    W8A8 format handler
    """
    def __init__(self, args: QuantArgs):
        super().__init__(args)
        
    def convert_linear(
        self,
        weight: torch.Tensor,
        qparams: Dict[str, torch.Tensor],
        qparams_key: str,
        dtype: torch.dtype,
    ):
        raise NotImplementedError()

    def pack_weight(
        self,
        weight: torch.Tensor,
        nrows: int,
        ncols: int
    ):
        raise NotImplementedError()
    
    def pack_params(
        self,
        scale: torch.Tensor,
        zero: torch.Tensor,
        nrows: int,
        ncols: int,
        group_size: int
    ):
        raise NotImplementedError()