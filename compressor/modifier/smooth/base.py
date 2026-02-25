import torch
import torch.nn as nn

from typing import *
from loguru import logger
from dataclasses import dataclass
from pydantic import PrivateAttr
from torch.utils.hooks import RemovableHandle

from compressor.calib import CalibDataLoader
from compressor.nn.struct import LLMStruct, DecoderStruct
from compressor.modifier.base import ModifierConfig, Modifier
from compressor.modifier.smooth.scale import compute_scale
from compressor.modifier.smooth.smooth import smooth_attn, smooth_quant
from compressor.utils import get_execution_device, ActStats, get_act_max, get_weight_max, log_weight

__all__ = ["SmoothConfig", "SmoothModifier"]

@dataclass
class SmoothConfig(ModifierConfig):
    """
    Configuration for SmoothQuant & SmoothAttn
    """
    path: str = ""

    # smoothattn
    smooth_attn: bool = True    # k proj -> q proj

    # smoothquant
    smooth_qkv: bool = True     # down proj -> qkv projs
    smooth_vo: bool = True      # v proj -> o proj
    smooth_ffn: bool = True     # o proj -> gate, up projs
    smooth_down: bool = True    # up, gate projs -> down proj

    # proj params
    proj_alpha: float = 0.5
    proj_beta: float = 0.5

    # attn params
    attn_alpha: float = 0.5
    attn_beta: float = 0.5

class SmoothModifier(Modifier):
    """
    SmoothQuant & SmoothAttn modifier

    1. Apply smoothattn to migrate keys' outliers to query
    2. Apply smoothquant to migrate activations' outliers to weight
    """
    config: SmoothConfig

    # runtime state
    _model_struct: Optional[LLMStruct] = PrivateAttr(default=None)
    _scales: Dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _smooth_hooks: Dict[int, Dict[str, RemovableHandle]] = PrivateAttr(default_factory=dict)
    _act_stats: Dict[int, Dict[str, ActStats]] = PrivateAttr(default_factory=dict)

    def initialize(self, model_struct: LLMStruct):
        """
        Initalize smooth modifier and register hooks for all layers
        """
        # initialize
        self._model_struct = model_struct
        self._scales = {}
        self._smooth_hooks = {}
        self._act_stats = {}

        # cached scales
        if self.config.path:
            self._scales = torch.load(self.config.path)
            self._initialized = True
            return

        # register all hooks
        for layer_idx, layer_struct in enumerate(model_struct.layer_structs):
            self._smooth_hooks[layer_idx] = {}
            self._act_stats[layer_idx] = {}

            # modules
            attn = layer_struct.attn_struct
            ffn = layer_struct.ffn_struct

            # for smoothattn
            if self.config.smooth_attn:
                self._register_hook(layer_idx, "k_proj_out", attn.k_proj)
                self._register_hook(layer_idx, "q_proj_out", attn.q_proj)

            # for smoothquant
            if self.config.smooth_qkv:
                self._register_hook(layer_idx, "q_proj", attn.q_proj, use_input=True)
            if self.config.smooth_vo:
                self._register_hook(layer_idx, "o_proj", attn.o_proj, use_input=True)
            if self.config.smooth_ffn:
                self._register_hook(layer_idx, "up_proj", ffn.up_proj, use_input=True)
            if self.config.smooth_down:
                self._register_hook(layer_idx, "down_proj", ffn.down_proj, use_input=True)

        self._initialized = True

    def _register_hook(
        self,
        layer_idx: int,
        name: str,
        module: nn.Module,
        use_input: bool = False,
    ):
        """
        Register forward hook with direct statistics collection
        """
        stat = ActStats()
        self._act_stats[layer_idx][name] = stat

        def hook_fn(_, inp, out):
            if use_input:
                x = inp[0] if isinstance(inp, tuple) else inp
            else:
                x = out[0] if isinstance(out, tuple) else out
            stat.update(x)

        handle = self.register_hook(module, hook_fn, "forward")
        self._smooth_hooks[layer_idx][name] = handle

    @torch.inference_mode()
    def apply(
        self, layer_struct: DecoderStruct, dataloader: CalibDataLoader
    ):
        """
        Apply smoothing transformations to a single decoder layer

        1. Collect activations via forward hooks
        2. SmoothAttn:
            k proj / s -> q proj * s
        3. SmoothQuant:
            input_norm / s -> qkv proj * s
            v proj / s -> o proj * s
            post norm / s -> gate, up proj * s
        """
        if not self._initialized:
            raise RuntimeError("SmoothModifier is not initialized")
        layer_idx = layer_struct.layer_idx

        # intialize structure
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct
        device = get_execution_device(layer_struct.module)

        # if cahced indices not exist, capture activations via hooks
        if layer_idx in self._smooth_hooks:
            # get current hooks
            layer_hooks = self._smooth_hooks[layer_idx]

            # collect activations via forward pass
            dataloader.calibrate(layer_struct, layer_hooks)
            layer_stats = self._act_stats[layer_idx]

            # remove stat hooks
            self.remove_hooks(set(layer_hooks.values()))
            self._smooth_hooks[layer_idx] = {}

        # smooth attention
        if self.config.smooth_attn:
            key = f"layer.{layer_idx}.attn"

            # compute scales
            if key not in self._scales:
                k_stats = layer_stats.get("k_proj_out")
                q_stats = layer_stats.get("q_proj_out")

                # get absmax with RoPE and GQA handling
                k_amax = get_act_max(k_stats, attn.config, is_kv=True, gqa=False)
                q_amax = get_act_max(q_stats, attn.config, is_kv=False, gqa=True)

                # compute scale at KV dimension
                scale = (k_amax ** self.config.attn_alpha) / (q_amax ** self.config.attn_beta)
                self._scales[key] = scale.cpu()

            # apply smoothing
            scale = self._scales[key].to(device)
            smooth_attn(attn.q_proj, attn.k_proj, scale, attn.config)
            
        # smooth input norm to qkv
        if self.config.smooth_qkv:
            key = f"layer.{layer_idx}.qkv"

            # compute scales
            if key not in self._scales:
                # get absolute max values
                amax = get_act_max(layer_stats.get("q_proj"))
                wmax = get_weight_max(attn.qkv_projs[0].weight)
                for p in attn.qkv_projs[1:]:
                    wmax = torch.maximum(wmax, get_weight_max(p.weight))
                
                # copmute scale
                scale = compute_scale(amax, wmax, self.config.proj_alpha, self.config.proj_beta)
                self._scales[key] = scale.cpu()
            
            # apply smoothing
            scale = self._scales[key].to(device)
            smooth_quant(layer_struct.input_layernorm, attn.qkv_projs, scale)

        # smooth v to o
        if self.config.smooth_vo:
            key = f"layer.{layer_idx}.vo"

            # compute scales
            if key not in self._scales:
                # get absolute max values
                amax = get_act_max(layer_stats.get("o_proj"), attn.config, is_kv=False, gqa=True)
                wmax = get_weight_max(attn.o_proj.weight, dim=0, attn_config=attn.config, reduce_gqa=True)

                # compute scale
                scale = compute_scale(amax, wmax, self.config.proj_alpha, self.config.proj_beta)
                self._scales[key] = scale.cpu()

            # apply smoothing
            scale = self._scales[key].to(device)            
            smooth_quant(attn.v_proj, attn.o_proj, scale)
            
        # smooth post norm to up projs
        if self.config.smooth_ffn:
            key = f"layer.{layer_idx}.ffn"

            # compute scales
            if key not in self._scales:
                # get absolute max values
                amax = get_act_max(layer_stats.get("up_proj"))
                wmax = get_weight_max(ffn.up_projs[0].weight)
                for p in ffn.up_projs[1:]:
                    wmax = torch.maximum(wmax, get_weight_max(p.weight))

                # compute scale
                scale = compute_scale(amax, wmax, self.config.proj_alpha, self.config.proj_beta)
                self._scales[key] = scale.cpu()
            
            # apply smoothing
            scale = self._scales[key].to(device)
            smooth_quant(layer_struct.post_attention_layernorm, ffn.up_projs, scale)

        # smooth up projs to down proj
        if self.config.smooth_down:
            key = f"layer.{layer_idx}.down"

            # compute scales
            if key not in self._scales:
                amax = get_act_max(layer_stats.get("down_proj"))
                wmax = get_weight_max(ffn.down_proj.weight)
                scale = compute_scale(amax, wmax, self.config.proj_alpha, self.config.proj_beta)
                self._scales[key] = scale.cpu()

            # apply smoothing
            scale = self._scales[key].to(device)
            smooth_quant(ffn.up_proj, ffn.down_proj, scale)

        # free memory
        del self._act_stats[layer_idx]

    def finalize(self):
        """
        Finalize after all layers processed

        1. Save computed scales
        2. Reset states
        """
        # save scales
        if self.config.path and self._scales:
            torch.save(self._scales, self.config.path)

        # reset states
        self._model_struct = None
        self._scales = {}
        self._smooth_hooks = {}
        self._act_stats = {}
        self._initialized = False

    def debug(self, layer_idx: int):
        """
        Log debug information
        """
        if self._model_struct is None or layer_idx < 0:
            return

        layer_struct = self._model_struct.layer_structs[layer_idx]
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct

        logger.debug(f"[SmoothModifier] Layer {layer_idx}:")

        # attention weights
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
            
        # scale
        for key in ["attn", "qkv", "vo", "ffn", "down"]:
            scale_key = f"layer.{layer_idx}.{key}"
            if scale_key in self._scales:
                s = self._scales[scale_key]
                logger.debug(
                    f"  scale[{key}]: mean={s.mean():.6f}, "
                    f"min={s.min():.6f}, max={s.max():.6f}"
                )
