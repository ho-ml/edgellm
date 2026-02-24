import gc
import torch

from typing import *
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizer
from compressor.config import CompressorConfig
from compressor.calib import CalibDataLoader
from compressor.nn import LLMStruct, DecoderStruct
from compressor.modifier import Modifier
from compressor.utils import (
    get_best_device, get_execution_device, log_weight, log_memory
)

__all__ = ["Compressor"]

# settings for logger
LEVEL = logger._core.min_level
DEBUG = 10

class Compressor:
    """
    Pipeline for block-wise compression of a LLM
    """
    def __init__(
        self,
        config: CompressorConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config

        # device
        self.offload_device = get_execution_device(model)
        self.onload_device = get_best_device()

        # build model struct
        self.model_struct = LLMStruct.from_model(model)
        self.model_struct.patch()

        # build dataloader
        self.dataloader: CalibDataLoader = config.calib.build_dataloader(
            model_struct=self.model_struct,
            tokenizer=tokenizer,
            offload_device=self.offload_device,
            onload_device=self.onload_device
        )

        # build modifiers
        self.modifiers: List[Modifier] = config.get_modifiers()

    def run(self):
        """
        Run full pipeline for a compression
        """
        # device
        logger.info(f"onload device: {self.onload_device}, offload device: {self.offload_device}")

        # initialize modifiers & dataloader
        model_struct = self.dataloader.model_struct
        for modifier in self.modifiers:
            logger.info(f"Initialize Modifier: {type(modifier).__name__}")
            modifier.initialize(model_struct)
        self.dataloader.initialize()
        logger.info(f"Initialized {len(self.modifiers)} modifiers successfully")
        self.clear()

        # layer-by-layer compression
        for layer_idx, layer_struct in self.model_struct.iter_layers():
            logger.info(f"Layer: {layer_idx} / {model_struct.num_layers - 1}")

            # move to best device
            layer_struct.module = layer_struct.module.to(self.onload_device)

            # debug before
            if LEVEL <= DEBUG:
                self.debug(layer_idx, layer_struct)

            # apply modifier
            for modifier in self.modifiers:
                logger.info(f"Apply Modifier: {type(modifier).__name__}")
                
                # clear the memory for debug
                if LEVEL <= DEBUG:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                # apply compression
                modifier.apply(layer_struct, self.dataloader)
                self.clear()

                # debug after
                if LEVEL <= DEBUG:
                    modifier.debug(layer_idx)
                    log_memory()

            # update hidden states for next layer
            logger.info(f"Finished Layer {layer_idx} successfully")                
            self.dataloader.propagate(layer_struct)

            # move to original device & clear memory
            layer_struct.module = layer_struct.module.to(self.offload_device)
            self.clear()

        # finalize all modifiers
        for modifier in self.modifiers:
            modifier.finalize()
        self.clear()

        return model_struct.model
    
    def clear(self):
        """
        Clear memory
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def debug(self, layer_idx: int, layer_struct: DecoderStruct):
        """
        Log debug information
        """
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct

        logger.debug(f"[Baseline] Layer {layer_idx}:")

        # attention weight
        log_weight("q_proj", attn.q_proj.weight)
        log_weight("k_proj", attn.k_proj.weight)
        log_weight("v_proj", attn.v_proj.weight)
        log_weight("o_proj", attn.o_proj.weight)

        # ffn weights
        if ffn.gate_proj is not None:
            log_weight("gate_proj", ffn.gate_proj.weight)
        log_weight("up_proj", ffn.up_proj.weight)
        log_weight("down_proj", ffn.down_proj.weight)
        
        # norm weights
        if layer_struct.input_layernorm is not None:
            log_weight("input_layernorm", layer_struct.input_layernorm.weight)
        if layer_struct.post_attention_layernorm is not None:
            log_weight("post_attention_layernorm", layer_struct.post_attention_layernorm.weight)