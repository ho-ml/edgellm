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
        return layer_struct.get_linear_modules()

