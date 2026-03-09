import torch

from typing import *
from loguru import logger
from compressor.nn.struct import LLMStruct
from compressor.config.quant import QuantConfig
from compressor.config.pack import PackConfig
from compressor.packer.format import infer_format
from compressor.packer.pack import build_state_dict, build_metadata, save_checkpoint

__all__ = ["Packer"]

class Packer:
    """
    Pack compressed model
    """
    def __init__(
        self,
        config: PackConfig,
        quant_config: QuantConfig,
        model_struct: LLMStruct,
        qparams: Dict[str, torch.Tensor]
    ):
        self.config = config
        self.quant_config = quant_config
        self.model_struct = model_struct
        self.qparams = qparams

        # infer format using quantization config
        self.format = infer_format(quant_config)

    def run(self):
        """
        Pack and save compressed model according to given format
        """
        logger.info(f"Packing model in {self.format} format")

        # resolve dtype
        dtype = getattr(torch, self.config.dtype)

        # get state dicts of compressed model
        args = self.quant_config.weight
        state_dict = build_state_dict(
            model_struct=self.model_struct,
            qparams=self.qparams,
            format_type=self.format,
            args=args,
            dtype=dtype,
        )

        # build metadata
        metadata = build_metadata(
            model_struct=self.model_struct,
            quant_config=self.quant_config,
            format_type=self.format,
            pack_config=self.config
        )

        # save checkpoints as safetensor
        filenames = save_checkpoint(
            state_dict=state_dict,
            metadata=metadata,
            output_dir=self.config.output_dir,
            max_shard_size=self.config.max_shard_size
        )
        logger.info(f"Checkpoint saved to {self.config.output_dir}: {filenames}")