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
            if logger._core.min_level <= DEBUG:
                self.debug(layer_idx, layer_struct)

            # apply modifier
            for modifier in self.modifiers:
                logger.info(f"Apply Modifier: {type(modifier).__name__}")
                
                # clear the memory for debug
                if logger._core.min_level <= DEBUG:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()

                # apply compression
                modifier.apply(layer_struct, self.dataloader)
                self.clear()

                # debug after
                if logger._core.min_level <= DEBUG:
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

if __name__ == "__main__":
    import sys
    import torch
    import argparse

    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from compressor.config import CompressorConfig, EvalConfig
    from compressor.evaluate import Evaluator

    # parse arguments
    parser = argparse.ArgumentParser(description="Compress a LLM")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument("--model",  required=True, help="Model ID or local path")
    parser.add_argument("--debug",  action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # set logger level
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    # parse config
    compress_config = CompressorConfig.from_yaml(args.config)
    eval_config = EvalConfig.from_yaml(args.config)

    # load models
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # evaluate baselines
    baseline_results = None
    if eval_config is not None and eval_config.compare_baseline:
        logger.info("Evaluating baseline model")
        baseline_results = Evaluator(eval_config, model, tokenizer).run()

    # compress
    compressor = Compressor(compress_config, model, tokenizer)
    compressed_model = compressor.run()

    # evaluate
    if eval_config is not None:
        Evaluator(eval_config, compressed_model, tokenizer).run(baseline_results)