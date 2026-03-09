import json
import torch

from typing import *
from pathlib import Path
from safetensors.torch import save_file
from compressor.nn import LLMStruct, DecoderStruct
from compressor.config.quant import QuantArgs, QuantConfig
from compressor.config.pack import PackConfig
from compressor.packer.format import FormatHandler

__all__ = ["build_state_dict", "build_metadata", "save_checkpoint"]

def _get_full_path(rname: str, layer_struct: DecoderStruct):
    """
    Map a rname to its full relateive path
    """
    if rname in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        return f"{layer_struct.attn_rname}.{rname}"
    elif rname in ["gate_proj", "up_proj", "down_proj"]:
        return f"{layer_struct.mlp_rname}.{rname}"
    
    return rname

def build_state_dict(
    model_struct: LLMStruct,
    qparams: Dict[str, torch.Tensor],
    format_type: str,
    args: QuantArgs,
    dtype: torch.dtype = torch.float16,
):
    """
    Build the complete packed state dict for the model
    """
    state_dict = {}

    # load format handler
    handler_cls = FormatHandler.get(format_type)
    handler = handler_cls(args)

    # embed layer
    if model_struct.embed_tokens is not None:
        embed_key = f"{model_struct.backbone_rname}.{model_struct.embed_tokens_rname}.weight"
        state_dict[embed_key] = model_struct.embed_tokens.weight.to(dtype).cpu()

    # layer-wise build
    for layer_idx, layer_struct in model_struct.iter_layers():
        prefix = f"{model_struct.backbone_rname}.{model_struct.layers_rname}.{layer_idx}"

        # normalization layers
        if layer_struct.input_layernorm is not None:
            key = f"{prefix}.{layer_struct.input_layernorm_rname}.weight"
            state_dict[key] = layer_struct.input_layernorm.weight.to(dtype).cpu()
        if layer_struct.post_attention_layernorm is not None:
            key = f"{prefix}.{layer_struct.post_attention_layernorm_rname}.weight"
            state_dict[key] = layer_struct.post_attention_layernorm.weight.to(dtype).cpu()
        
        # quantized linear layers
        linears = layer_struct.get_linear_modules()
        for rname, module in linears.items():
            # get prefix
            qparams_key = f"layer.{layer_idx}.{rname}"
            path = _get_full_path(rname, layer_struct)
            full_prefix = f"{prefix}.{path}"

            # convert linear to qlinear
            qlinear = handler.convert_linear(
                module.weight, qparams, qparams_key, dtype=dtype
            )

            # save to state dict
            for suffix, tensor in qlinear.items():
                full_key = f"{full_prefix}.{suffix}"
                state_dict[full_key] = tensor

    # final normalization
    if model_struct.norm is not None:
        norm_key = f"{model_struct.backbone_rname}.{model_struct.norm_rname}.weight"
        state_dict[norm_key] = model_struct.norm.weight.to(dtype).cpu()

    # lm head
    if model_struct.lm_head is not None:
        state_dict[f"{model_struct.lm_head_rname}.weight"] = model_struct.lm_head.weight.to(dtype).cpu()

    return state_dict

def build_metadata(
    model_struct: LLMStruct,
    quant_config: QuantConfig,
    format_type: str,
    pack_config: PackConfig,
):
    """
    Build config.json metadata
    """
    model_config = model_struct.config

    # quantization meta data
    quant_meta = {
        "weight": quant_config.weight.bits,
    }
    if quant_config.input is not None:
        quant_meta["input"] = quant_config.input.bits
    if quant_config.output is not None:
        quant_meta["output"] = quant_config.output.bits

    # meta data
    meta = {
        "format": format_type,
        "dtype": pack_config.dtype,
    
        "architecture": {
            "hidden_size": model_config.hidden_size,
            "num_hidden_layers": model_config.num_hidden_layers,
            "num_attention_heads": model_config.num_heads,
            "num_key_value_heads": model_config.kv_heads,
            "intermediate_size": model_config.intermediate_size,
            "vocab_size": model_config.vocab_size,
        },

        "quantization": quant_meta,
        "symmetric": quant_config.weight.symmetric,

        "engine": {
            "target_sm": 89,        # RTX 4060
        }
    }

    return meta

def _split_shard(
    state_dict: Dict[str, torch.Tensor],
    max_bytes: int
):
    """
    Split state dict into shards
    """
    # initialize
    shards = []
    cur_shard = {}
    cur_size = 0

    for key, tensor in state_dict.items():
        # calculate tensor memory
        tensor_size = tensor.nelement() * tensor.element_size()

        # save current shard and make new
        if cur_size + tensor_size > max_bytes and cur_shard:
            shards.append(cur_shard)
            cur_shard = {}
            cur_size = 0

        # update current shard
        cur_shard[key] = tensor
        cur_size += tensor_size

    # append remaining shard
    if cur_shard:
        shards.append(cur_shard)

    return shards


def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict,
    output_dir: str,
    max_shard_size: float = 4.0
):
    """
    Save state dict as safetensors & config.json
    """
    # initialize
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_shard_size * 1024 ** 3)
    
    # split into shards
    shards = _split_shard(state_dict, max_bytes)
    total = len(shards)
    
    # save to safetensors
    filenames = []
    for i, shard in enumerate(shards):
        filename = f"model-{i+1:05d}-of-{total:05d}.safetensors"
        filepath = output_path / filename

        # save
        save_file(shard, str(filepath))
        filenames.append(filename)

    # add file list to metadata
    metadata["files"] = filenames

    # save config.json
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return filenames
