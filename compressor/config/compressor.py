import yaml

from typing import *
from loguru import logger
from dataclasses import dataclass, field
from compressor.config.calib import CalibConfig
from compressor.config.quant import QuantConfig
from compressor.modifier.rotate import RotateConfig, RotateModifier
from compressor.modifier.smooth import SmoothConfig, SmoothModifier
from compressor.modifier.reorder import ReorderConfig, ReorderModifier
from compressor.modifier.weight import WeightQuantConfig, WeightQuantModifier

__all__ = ["CompressorConfig"]


@dataclass
class CompressorConfig:
    """
    Configuration for the compression pipeline
    """
    # Config for calibration
    calib: CalibConfig

    # Config for specific modifiers
    rotate: Optional[RotateConfig] = field(default=None)
    reorder: Optional[ReorderConfig] = field(default=None)
    smooth: Optional[SmoothConfig] = field(default=None)

    # Config for quantization
    quant: Optional[QuantConfig] = field(default=None)
    weight_quant: Optional[WeightQuantConfig] = field(default=None)

    def __post_init__(self):
        """
        Validate and adjust config
        """
        rotate_enabled = self.rotate is not None and self.rotate.enabled
        reorder_enabled = self.reorder is not None and self.reorder.enabled

        # rotate vs. reorder
        if rotate_enabled and reorder_enabled:
            # rotate_down + reorder_down
            if self.rotate.rotate_down and self.reorder.reorder_down:
                logger.warning("rotate_down and reorder_down are incompatible.")
                self.reorder.reorder_down = False

            # rotate_out + reorder_out
            if self.rotate.rotate_out and self.reorder.reorder_out:
                logger.warning("rotate_out and reorder_out are incompatible.")
                self.reorder.reorder_out = False 

    def get_modifiers(self):
        """
        Return list of enabled modifiers in pipeline order:
        rotate -> reorder -> smooth -> weight quant
        """
        modifiers = []

        # Rotate modifier
        if self.rotate is not None and self.rotate.enabled:
            modifier = RotateModifier(config=self.rotate)
            modifiers.append(modifier)

        # Reorder modifier
        if self.reorder is not None and self.reorder.enabled:
            modifier = ReorderModifier(config=self.reorder)
            modifiers.append(modifier)

        # Smooth modifier
        if self.smooth is not None and self.smooth.enabled:
            modifier = SmoothModifier(config=self.smooth)
            modifiers.append(modifier)

        # Weight quantization modifier
        if self.weight_quant is not None:
            modifier = WeightQuantModifier(config=self.weight_quant)
            modifiers.append(modifier)
        
        return modifiers

    @classmethod
    def from_yaml(cls, path: str):
        """
        Load configuration from YAML file
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Calib config
        calib_kwargs = config.get("calib", {})
        calib_config = CalibConfig(**calib_kwargs)

        # Rotate config
        rotate_kwargs = config.get("rotate", None)
        if rotate_kwargs is not None:
            rotate_config = RotateConfig(**rotate_kwargs)
        else:
            rotate_config = None

        # Reorder config
        reorder_kwargs = config.get("reorder", None)
        if reorder_kwargs is not None:
            reorder_config = ReorderConfig(**reorder_kwargs)
        else:
            reorder_config = None

        # Smooth config
        smooth_kwargs = config.get("smooth", None)
        if smooth_kwargs is not None:
            smooth_config = SmoothConfig(**smooth_kwargs)
        else:
            smooth_config = None

        # Quantization config
        quant_kwargs = config.get("quant", None)
        if quant_kwargs is not None:
            quant_config = QuantConfig.from_dict(quant_kwargs)
        else:
            quant_config = None

        # Weight quantization config
        weight_quant_kwargs = config.get("weight_quant")
        if weight_quant_kwargs is not None:
            weight_quant_config = WeightQuantConfig(**weight_quant_kwargs)
            weight_quant_config.args = quant_config.weight
        else:
            weight_quant_config = None

        return cls(
            calib=calib_config,
            rotate=rotate_config,
            reorder=reorder_config,
            smooth=smooth_config,
            quant=quant_config,
            weight_quant=weight_quant_config
        )