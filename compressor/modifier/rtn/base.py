import torch

from typing import *
from pydantic import PrivateAttr

from compressor.calib import CalibDataLoader
from compressor.observers import Observer
from compressor.nn.struct import LLMStruct, DecoderStruct
from compressor.modifier.weight.base import WeightQuantConfig
from compressor.modifier.base import Modifier
from compressor.utils import fake_quantize
from compressor.modifier.weight.progressive import calculate_progressive_scales, restore_weight

__all__ = ["RTNModifier"]

class RTNModifier(Modifier):
    """
    RTN (Round-to-Nearest) weight quantization modifier
    """
    config: WeightQuantConfig

    # runtime state
    _model_struct: Optional[LLMStruct] = PrivateAttr(default=None)

    def initialize(self, model_struct: LLMStruct):
        """
        Initialize RTN modifier
        """
        self._model_struct = model_struct

    @torch.no_grad()
    def apply(
        self, layer_struct: DecoderStruct, _dataloader: CalibDataLoader
    ):
        """
        Apply RTN quantization to a decoder layer
        """
        linears = self.get_linear_modules(layer_struct)

        for _, module in linears.items():
            W = module.weight.clone().to(torch.float32)
            ori_dtype = module.weight.dtype
            args = self.config.args

            if args.is_progressive:
                # compute 2-level scales
                x_norm, scale_0, scale_1, zero_1 = calculate_progressive_scales(W, args)
                level0_col = args.group_shapes[0][-1]

                # fake quantize normalized weight
                qweight = fake_quantize(x_norm, scale_1, zero_1, args)
                qweight = restore_weight(qweight, scale_0, level0_col).to(ori_dtype)
                
            else:
                # get observer and compute scales
                observer_cls = Observer.get(args.observer)
                observer = observer_cls(args=args)
                scale, zero = observer(W, target="weight")

                # apply fake quantize
                qweight = fake_quantize(W, scale, zero, args)
                qweight = qweight.to(ori_dtype)

            module.weight.data = qweight

    def finalize(self):
        """
        Finalize after all layers processed
        """
        self._model_struct = None

    def debug(self, layer_idx: int):
        """
        Log debug information
        """
        pass
