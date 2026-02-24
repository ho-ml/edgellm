import torch
import torch.nn as nn

from typing import *
from loguru import logger
from pydantic import PrivateAttr
from torch.utils.hooks import RemovableHandle

from compressor.calib import CalibDataLoader
from compressor.nn.struct import LLMStruct, DecoderStruct
from compressor.modifier.weight.base import WeightQuantConfig
from compressor.modifier.base import Modifier
from compressor.modifier.gptq.gptq import quantize_weight
from compressor.modifier.weight.progressive import restore_weight
from compressor.modifier.gptq.hessian import (
    init_hessian, accumulate_hessian, onload_hessian
)

__all__ = ["GPTQModifier"]

class GPTQModifier(Modifier):
    """
    GPTQ weight quantization modifier
    """
    config: WeightQuantConfig

    # runtime state
    _model_struct: Optional[LLMStruct] = PrivateAttr(default=None)
    _gptq_hooks: Dict[int, Dict[str, RemovableHandle]] = PrivateAttr(default_factory=dict)
    _hessians: Dict[int, Dict[str, torch.Tensor]] = PrivateAttr(default_factory=dict)
    _num_samples: Dict[int, Dict[str, int]] = PrivateAttr(default_factory=dict)
    _qparams: Dict[str, torch.Tensor | None] = PrivateAttr(default_factory=dict)
    _losses: Dict[int, Dict[str, float]] = PrivateAttr(default_factory=dict)

    def initialize(self, model_struct: LLMStruct):
        """
        Initialize states and register hooks for all layers
        """
        self._model_struct = model_struct
        self._gptq_hooks = {}
        self._hessians = {}
        self._num_samples = {}
        self._qparams = {}
        self._losses = {}

        for layer_idx, layer_struct in enumerate(model_struct.layer_structs):
            self._gptq_hooks[layer_idx] = {}
            self._hessians[layer_idx] = {}
            self._num_samples[layer_idx] = {}

            # get all linears
            linears = self.get_linear_modules(layer_struct)
            for n, m in linears.items():
                self._register_hook(layer_idx, n, m)

    def _register_hook(
        self, layer_idx: int, name: str, module: nn.Module
    ):
        """
        Register forward hook for Hessian collection
        """
        def hook_fn(_, inp, _out):
            layer_hessian = self._hessians[layer_idx]
            layer_samples = self._num_samples[layer_idx]

            # initialize hessian
            if name not in layer_hessian:
                layer_hessian[name] = init_hessian(module)
                layer_samples[name] = 0

            # accumulate hessian
            x = inp[0] if isinstance(inp, tuple) else inp
            with onload_hessian(name, module, layer_hessian):
                layer_hessian[name], layer_samples[name] = accumulate_hessian(
                    x, layer_hessian[name], layer_samples[name]
                )

        handle = self.register_hook(module, hook_fn, "forward")
        self._gptq_hooks[layer_idx][name] = handle

    @torch.inference_mode()
    def apply(
        self, layer_struct: DecoderStruct, dataloader: CalibDataLoader
    ):
        """
        Apply GPTQ quantization to a decoder layer

        1. Collect activations via forward hooks (Hessian computation)
        2. Quantize all linear modules which are in this layer
        """
        layer_idx = layer_struct.layer_idx

        # collect activations
        layer_hooks = self._gptq_hooks[layer_idx]
        dataloader.calibrate(layer_struct, layer_hooks)

        # remove hooks
        self.remove_hooks(set(layer_hooks.values()))
        self._gptq_hooks[layer_idx] = {}

        # current layer info
        linears = self.get_linear_modules(layer_struct)
        layer_hessians = self._hessians.get(layer_idx, {})

        # initialize losses for this layer
        self._losses[layer_idx] = {}

        # quantize each linear module
        for name, module in linears.items():
            with onload_hessian(name, module, layer_hessians):
                result = quantize_weight(
                    name=name,
                    module=module,
                    args=self.config.args,
                    hessians_dict=layer_hessians,
                    block_size=self.config.block_size,
                    perc_damp=self.config.perc_damp,
                )

            if self.config.args.is_progressive:
                loss, qweight, scale_0, scale_1, zero = result
                level0_col = self.config.args.group_shapes[0][-1]

                # restore float weight using level-0 scale
                restored = restore_weight(qweight, scale_0, level0_col)
                module.weight.data = restored.to(qweight.dtype)
                self._qparams[f"layer.{layer_idx}.{name}.scale_0"] = scale_0
                self._qparams[f"layer.{layer_idx}.{name}.scale_1"] = scale_1
                self._qparams[f"layer.{layer_idx}.{name}.zero"] = zero
            
            else:
                loss, qweight, scale, zero = result
                module.weight.data = qweight
                self._qparams[f"layer.{layer_idx}.{name}.scale"] = scale
                self._qparams[f"layer.{layer_idx}.{name}.zero"] = zero

            # save result
            self._losses[layer_idx][name] = loss

        # clean layer dicts
        if layer_idx in self._hessians:
            del self._hessians[layer_idx]
        if layer_idx in self._num_samples:
            del self._num_samples[layer_idx]
    
    def finalize(self):
        """
        Finalize after all layers processed

        1. save quantized params
        2. reset states
        """
        # save params
        if self.config.path and self._qparams:
            torch.save(self._qparams, self.config.path)

        # reset states
        self._model_struct = None
        self._gptq_hooks = {}
        self._hessians = {}
        self._num_samples = {}
        self._qparams = {}
        self._losses = {}

    def debug(self, layer_idx: int):
        """
        Log debug information
        """
        if self._model_struct is None or layer_idx not in self._losses:
            return
        layer_losses = self._losses[layer_idx]

        logger.debug(f"[GPTQModifier] Layer {layer_idx}:")

        # module loss
        for name, loss in layer_losses.items():
            logger.debug(f"  {name}: loss={loss:.6f}")

            
            