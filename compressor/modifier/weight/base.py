from typing import *
from dataclasses import dataclass, field
from pydantic import PrivateAttr

from compressor.config import QuantArgs
from compressor.calib import CalibDataLoader
from compressor.nn.struct import LLMStruct, DecoderStruct
from compressor.modifier.base import Modifier

__all__ = ["WeightQuantConfig", "WeightQuantModifier"]


@dataclass
class WeightQuantConfig:
    """
    Configuration for weight quantization
    """
    args: QuantArgs = field(default_factory=QuantArgs)

    path: str = ""
    method: str = "gptq"    # "gptq", "rtn", "awq"
    block_size: int = 128
    perc_damp: float = 0.01


class WeightQuantModifier(Modifier):
    """
    Weight quantization modifier

    - call specific modifiers based on method:
        - "gptq": GPTQModifier
        - "rtn": RTNModifier
        - "awq": AWQModifier
    - supports progressive quantization
    """
    config: WeightQuantConfig

    # internal modifier instance
    _modifier: Optional[Modifier] = PrivateAttr(default=None)

    def initialize(self, model_struct: LLMStruct):
        """
        Initialize by creating and initializing the appropriate modifier
        """
        method = self.config.method.lower()
        
        # create internal modifier
        if method == "gptq":
            from compressor.modifier.gptq import GPTQModifier
            self._modifier = GPTQModifier(config=self.config)

        elif method == "rtn":
            from compressor.modifier.rtn import RTNModifier
            self._modifier = RTNModifier(config=self.config)

        elif method == "awq":
            raise NotImplementedError("AWQ method is not yet implemented")

        else:
            raise ValueError(f"Unknown quantization method: {method}")

        # initialize for specific modifiers
        self._modifier.initialize(model_struct)
        self._initialized = True

    def apply(
        self, layer_struct: DecoderStruct, dataloader: CalibDataLoader
    ):
        """
        Apply quantization using internal modifier
        """
        self._modifier.apply(layer_struct, dataloader)

    def finalize(self):
        """
        Finalize by cleaning up the internal modifier
        """
        if self._modifier is not None:
            self._modifier.finalize()
            self._modifier = None

    def debug(self, layer_idx: int):
        """
        Debug to internal modifier
        """
        if self._modifier is not None:
            self._modifier.debug(layer_idx)
