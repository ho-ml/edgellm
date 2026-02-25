import os
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
from compressor.modifier.activation import ActivationQuantConfig, ActivationQuantModifier

__all__ = ["CompressorConfig"]

@dataclass
class CompressorConfig:
    """
    Configuration for the compression pipeline
    """
    # config for calibration
    calib: CalibConfig

    # config for specific modifiers
    transform_path: str = ""
    rotate: Optional[RotateConfig] = field(default=None)
    reorder: Optional[ReorderConfig] = field(default=None)
    smooth: Optional[SmoothConfig] = field(default=None)

    # config for quantization
    quant_path: str = ""
    quant: Optional[QuantConfig] = field(default=None)
    weight_quant: Optional[WeightQuantConfig] = field(default=None)
    act_quant: Optional[ActivationQuantConfig] = field(default=None)

    def __post_init__(self):
        """
        Validate and adjust config
        """
        # rotate vs. reorder
        rotate_enabled = self.rotate is not None and self.rotate.enabled
        reorder_enabled = self.reorder is not None and self.reorder.enabled
        if rotate_enabled and reorder_enabled:
            # rotate_down + reorder_down
            if self.rotate.rotate_down and self.reorder.reorder_down:
                logger.warning("rotate_down and reorder_down are incompatible.")
                self.reorder.reorder_down = False

            # rotate_out + reorder_out
            if self.rotate.rotate_out and self.reorder.reorder_out:
                logger.warning("rotate_out and reorder_out are incompatible.")
                self.reorder.reorder_out = False

        # smooth proj alpha
        if self.smooth is not None and self.smooth.enabled:
            if abs(self.smooth.proj_alpha + self.smooth.proj_beta - 1.0) > 1e-6:
                logger.warning("sum of sooth alpha and smooth beta must be 1.")
                self.smooth.proj_beta = 1 - self.smooth.proj_alpha

        # determine paths from the transform_path
        if self.transform_path:
            if self.rotate is not None and not self.rotate.path:
                self.rotate.path = os.path.join(self.transform_path, "rotate")
            if self.reorder is not None and not self.reorder.path:
                self.reorder.path = os.path.join(self.transform_path, "reorder")
            if self.smooth is not None and not self.smooth.path:
                self.smooth.path = os.path.join(self.transform_path, "smooth")

        # determine paths from the quant_path
        if self.quant_path:
            if self.weight_quant is not None and not self.weight_quant.path:
                self.weight_quant.path = os.path.join(self.quant_path, "weight")
        
        # unify `skips` types to list
        if self.act_quant is not None and isinstance(self.act_quant.skips, str):
            self.act_quant.skips = [self.act_quant.skips]

        # create parent directories for all output paths
        for path in [
            self.rotate.path if self.rotate is not None else "",
            self.reorder.path if self.reorder is not None else "",
            self.smooth.path if self.smooth is not None else "",
            self.weight_quant.path if self.weight_quant is not None else "",
        ]:
            if path:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)

    def get_modifiers(self):
        """
        Return list of enabled modifiers in pipeline order:
        rotate -> reorder -> smooth -> activation quant -> weight quant
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

        # Activation quantization modifier
        if self.act_quant is not None:
            modifier = ActivationQuantModifier(config=self.act_quant)
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

        # Transform config
        transform_dict = config.get("transform", None)
        transform_path = ""
        rotate_config = None
        reorder_config = None
        smooth_config = None

        if transform_dict:
            transform_path = transform_dict.get("path", "")

            rotate_kwargs = transform_dict.get("rotate", None)
            if rotate_kwargs is not None:
                rotate_config = RotateConfig(**rotate_kwargs)

            reorder_kwargs = transform_dict.get("reorder", None)
            if reorder_kwargs is not None:
                reorder_config = ReorderConfig(**reorder_kwargs)

            smooth_kwargs = transform_dict.get("smooth", None)
            if smooth_kwargs is not None:
                smooth_config = SmoothConfig(**smooth_kwargs)

        # quant config
        quant_dict = config.get("quant", None)
        quant_path = ""
        quant_config = None
        weight_quant_config = None
        act_quant_config = None

        if quant_dict:
            quant_path = quant_dict.get("path", "")
            quant_config = QuantConfig.from_dict(quant_dict)

            # weight quantization config
            weight_dict = quant_dict.get("weight", None)
            if weight_dict is not None and quant_config.weight is not None:
                mod_dict = weight_dict.get("mod", {})
                weight_quant_config = WeightQuantConfig(
                    args=quant_config.weight,
                    method=mod_dict.get("method", "gptq"),
                    block_size=mod_dict.get("block_size", 128),
                    perc_damp=mod_dict.get("perc_damp", 0.01)
                )

            # activation quantization config
            if quant_config.input is not None or quant_config.output is not None:
                output_dict = quant_dict.get("output", None)
                if output_dict is not None:
                    output_mod = output_dict.get("mod", {})
                    skips = output_mod.get("skips", [])

                act_quant_config = ActivationQuantConfig(
                    input_args=quant_config.input,
                    output_args=quant_config.output,
                    skips=skips if skips else []
                )

        return cls(
            calib=calib_config,
            transform_path=transform_path,
            rotate=rotate_config,
            reorder=reorder_config,
            smooth=smooth_config,
            quant_path=quant_path,
            quant=quant_config,
            weight_quant=weight_quant_config,
            act_quant=act_quant_config
        )
