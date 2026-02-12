from typing import *
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pydantic import PrivateAttr
from compressor.utils import HooksMixin
from compressor.calib import CalibDataLoader
from compressor.nn.struct import LLMStruct, DecoderStruct

__all__ = ["ModifierConfig", "Modifier"]

@dataclass
class ModifierConfig:
    """
    Base configuration for modifiers
    """
    enabled: bool = False

class Modifier(HooksMixin, ABC):
    """
    Base class for all compression modifiers
    """
    config: ModifierConfig

    # state
    _initialized: bool = PrivateAttr(default=False)

    @abstractmethod
    def initialize(self, model_struct: LLMStruct):
        """
        Initialize modifiers
        """
        ...

    @abstractmethod
    def apply(
        self, layer_struct: DecoderStruct, dataloader: CalibDataLoader
    ):
        """
        Apply modifiers to specific decoder layer
        """
        ...

    @abstractmethod
    def finalize(self):
        """
        Finalize modifiers
        """
        ...

    @abstractmethod
    def debug(self, layer_idx: int):
        """
        Create logs for debug
        """
        ...
    
    def get_linear_modules(self, layer_struct: DecoderStruct):
        """
        Get all linear modules for a specific layer
        """
        linears = {}
        
        # modules
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct

        # attention module
        if hasattr(attn, "q_proj") and attn.q_proj is not None:
            linears[attn.q_proj_rname] = attn.q_proj
        if hasattr(attn, "k_proj") and attn.k_proj is not None:
            linears[attn.k_proj_rname] = attn.k_proj
        if hasattr(attn, "v_proj") and attn.v_proj is not None:
            linears[attn.v_proj_rname] = attn.v_proj
        if hasattr(attn, "o_proj") and attn.o_proj is not None:
            linears[attn.o_proj_rname] = attn.o_proj

        # ffn module
        if hasattr(ffn, "gate_proj") and ffn.gate_proj is not None:
            linears[ffn.gate_proj_rname] = ffn.gate_proj
        if hasattr(ffn, "up_proj") and ffn.up_proj is not None:
            linears[ffn.up_proj_rname] = ffn.up_proj
        if hasattr(ffn, "down_proj") and ffn.down_proj is not None:
            linears[ffn.down_proj_rname] = ffn.down_proj

        return linears

        
