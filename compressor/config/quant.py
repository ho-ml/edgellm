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

    # observer
    observer: str = "memoryless-minmax"
    observer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # params dtype
    zero_dtype: Optional[str] = None
    group_shapes: Optional[List[List[int]]] = None 
    scale_dtypes: Optional[List[str]] = None

    # progressive quantization
    intermediate_bits: Optional[int] = None

    # zero point domain
    zero_domain: Optional[str] = None

    @property
    def is_progressive(self):
        return (
            self.group_shapes is not None 
            and len(self.group_shapes) > 1
            and self.intermediate_bits is not None
        )

@dataclass
class QuantConfig:
    """
    Quantization config for compressor
    """
    weight: Optional[QuantArgs] = None
    input: Optional[QuantArgs] = None
    output: Optional[QuantArgs] = None

    @classmethod
    def w4a8(cls):
        """
        W4A8 preset: 4-bit weight, 8-bit input activation
        """
        return cls(
            weight=QuantArgs(
                bits=4,
                strategy="group",
                group_shapes=[[1, 128]],
                observer="memoryless-minmax"
            ),
            input=QuantArgs(
                bits=8,
                strategy="token",
                dynamic=True,
                observer="minmax"
            ),
            output=QuantArgs(
                bits=4,
                symmetric=False,
                strategy="group",
                group_shapes=[[1, 128]],
                dynamic=True,
                observer="minmax"
            ),
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
                group_shapes=[[1, 128]],
                observer="memoryless-minmax"
            ),
            output=QuantArgs(
                bits=4,
                symmetric=False,
                strategy="group",
                group_shapes=[[1, 128]],
                dynamic=True,
                observer="minmax"
            ),
        )

    @classmethod
    def w8a8(cls):
        """
        W8A8 preset: 8-bit weight, 8-bit input activation quantization
        """
        return cls(
            weight=QuantArgs(
                bits=8,
                strategy="channel",
                observer="memoryless-minmax"
            ),
            input=QuantArgs(
                bits=8,
                strategy="token",
                dynamic=True,
                observer="minmax"
            ),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Create config from dictionary
        """
        weight = None
        input_args = None
        output_args = None

        # weight
        weight_dict = config_dict.get("weight", None)
        if weight_dict is not None:
            args_dict = weight_dict.get("args", {})
            if args_dict:
                weight = QuantArgs(**args_dict)

        # input
        input_dict = config_dict.get("input", None)
        if input_dict is not None:
            args_dict = input_dict.get("args", {})
            if args_dict:
                input_args = QuantArgs(**args_dict)

        # output
        output_dict = config_dict.get("output", None)
        if output_dict is not None:
            args_dict = output_dict.get("args", {})
            if args_dict:
                output_args = QuantArgs(**args_dict)

        return cls(
            weight=weight,
            input=input_args,
            output=output_args,
        )
