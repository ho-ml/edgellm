from typing import *
from dataclasses import dataclass, field

__all__ = ["QuantArgs", "QuantConfig"]

@dataclass
class QuantArgs:
    """
    Quantization args for a specific target
    """
    bits: int = 8
    symmetric: bool = True
    strategy: str = "channel"   # tensor, channel, group, token
    dynamic: bool = False
    group_size: int = 128

    # observer
    observer: str = "memoryless-minmax"
    observer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # params dtype
    scale_dtype: Optional[str] = None   
    zero_dtype: Optional[str] = None

@dataclass
class QuantConfig:
    """
    Quantization config for compressor
    """
    weight: Optional[QuantArgs] = None
    activation: Optional[QuantArgs] = None
    kv_cache: Optional[QuantArgs] = None

    @classmethod
    def w4a8(cls):
        """
        W4A8 preset: 4-bit weight, 8-bit activation
        """
        return cls(
            weight=QuantArgs(
                bits=4,
                strategy="group",
                group_size=128,
                observer="memoryless-minmax"
            ),
            activation=QuantArgs(
                bits=8,
                strategy="token",
                dynamic=True,
                observer="minmax"
            ),
            kv_cache=None
        )

    @classmethod
    def w4a16(cls):
        """
        W4A16 preset: 4-bit weight, no activation quantization
        """
        return cls(
            weight=QuantArgs(
                bits=4,
                strategy="group",
                group_size=128,
                observer="memoryless-minmax"
            ),
            activation=None,
            kv_cache=None
        )

    @classmethod
    def w8a8(cls):
        """
        W8A8 preset: 8-bit weight, 8-bit activation quantization
        """
        return cls(
            weight=QuantArgs(
                bits=8,
                strategy="channel",
                group_size=128,
                observer="memoryless-minmax"
            ),
            activation=QuantArgs(
                bits=8,
                strategy="token",
                dynamic=True,
                observer="minmax"
            ),
            kv_cache=None
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Create config from dictionary
        """
        weight = None
        activation = None
        kv_cache = None

        # quant config
        if "weight" in config_dict and config_dict["weight"] is not None:
            weight = QuantArgs(**config_dict["weight"])
        if "activation" in config_dict and config_dict["activation"] is not None:
            activation = QuantArgs(**config_dict["activation"])
        if "kv_cache" in config_dict and config_dict["kv_cache"] is not None:
            kv_cache = QuantArgs(**config_dict["kv_cache"])
                
        return cls(
            weight=weight, 
            activation=activation, 
            kv_cache=kv_cache
        )
