import torch

from typing import *
from loguru import logger
from dataclasses import dataclass, field
from pydantic import PrivateAttr

from compressor.config.quant import QuantArgs
from compressor.calib import CalibDataLoader
from compressor.nn.struct import LLMStruct, DecoderStruct
from compressor.modifier.base import Modifier
from compressor.modifier.activation.hooks import DynamicQuantHook

__all__ = ["ActivationQuantConfig", "ActivationQuantModifier"]

@dataclass
class ActivationQuantConfig:
    """
    Configuration for activation quantization
    """
    input_args: Optional[QuantArgs] = field(default=None)
    output_args: Optional[QuantArgs] = field(default=None)
    skips: List[str] = field(default_factory=list)

class ActivationQuantModifier(Modifier):
    """
    Activation quantization modifier

    - input: per-token dynamic quantization on linear inputs
    - output: group-wise dynamic quantization on attention outputs
    """
    config: ActivationQuantConfig

    # runtime states
    _model_struct: Optional[LLMStruct] = PrivateAttr(default=None)

    def initialize(self, model_struct: LLMStruct):
        self._model_struct = model_struct
        self._initialized = True

    @torch.inference_mode()
    def apply(
        self, layer_struct: DecoderStruct, _dataloader: CalibDataLoader
    ):
        """
        Apply dynamic quantization for activations
        
        1. Register forward_pre hooks for input activation
        2. Register forward hooks for output activation
        2. Quantize activations in real time (online)
        """
        if not self._initialized:
            raise RuntimeError("ActivationQuantModifier is not initialized")

        # get infos
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct
        skips = set(self.config.skips)
        input_args = self.config.input_args
        output_args = self.config.output_args

        # qkv projs
        for rname in [attn.q_proj_rname, attn.k_proj_rname, attn.v_proj_rname]:
            proj = getattr(attn, rname, None)
            if proj is None: continue

            # input
            if input_args is not None:
                hook = DynamicQuantHook(args=input_args)
                self.register_persistent_hook(
                    proj, hook.as_input_hook(),
                    "forward_pre", with_kwargs=True
                )

            # output
            if output_args is not None and rname not in skips:
                hook = DynamicQuantHook(args=output_args)
                self.register_persistent_hook(
                    proj, hook.as_output_hook(), "forward"
                )

        # o proj
        if input_args is not None and attn.o_proj is not None:
            hook = DynamicQuantHook(args=input_args)
            self.register_persistent_hook(
                attn.o_proj, hook.as_input_hook(),
                "forward_pre", with_kwargs=True
            )

        # up projs
        if input_args is not None:
            for proj in ffn.up_projs:
                hook = DynamicQuantHook(args=input_args)
                self.register_persistent_hook(
                    proj, hook.as_input_hook(),
                    "forward_pre", with_kwargs=True
                )

            # down proj
            hook = DynamicQuantHook(args=input_args)
            self.register_persistent_hook(
                ffn.down_proj, hook.as_input_hook(),
                "forward_pre", with_kwargs=True
            )
    
    def finalize(self):
        """
        Finalize after all layers processed
        """
        self._model_struct = None
        self._initialized = False
    
    def debug(self, layer_idx: int):
        """
        Log debug information
        """
        ...