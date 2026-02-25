import os
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
from compressor.modifier.reorder.reorder import (
    compute_index, reorder, compute_head_index
)
from compressor.utils import ActStats

DEBUG = 10

@dataclass
class ReorderConfig(ModifierConfig):
    """
    Channel reordering configuration
    """
    path: str = ""

    reorder_out: bool = True 
    reorder_down: bool = True

class ReorderModifier(Modifier):
    """
    Channel reordering modifier for improved group quantization
    """
    config: ReorderConfig

    # runtime state
    _model_struct: Optional[LLMStruct] = PrivateAttr(default=None)
    _indices: Dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _reorder_hooks: Dict[int, Dict[str, RemovableHandle]] = PrivateAttr(default_factory=dict)
    _act_stats: Dict[int, Dict[str, ActStats]] = PrivateAttr(default_factory=dict)
    _debug_info: Dict[int, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def initialize(self, model_struct: LLMStruct):
        """
        Initialize reorder modifier
        """
        self._model_struct = model_struct
        self._indices = {}
        self._reorder_hooks = {}
        self._act_stats = {}
        self._debug_info = {}

        # cached indices
        if self.config.path and os.path.isfile(self.config.path):
            self._indices = torch.load(self.config.path)
            self._initialized = True
            return
        
        # register all hooks
        for layer_idx, layer_struct in enumerate(model_struct.layer_structs):
            self._reorder_hooks[layer_idx] = {}
            self._act_stats[layer_idx] = {}
            if logger._core.min_level <= DEBUG:
                self._debug_info[layer_idx] = {}
            
            # get modules
            attn = layer_struct.attn_struct
            ffn = layer_struct.ffn_struct

            # hooks
            if self.config.reorder_out:
                self._register_hook(layer_idx, "o_proj", attn.o_proj)
            if self.config.reorder_down:
                self._register_hook(layer_idx, "down_proj", ffn.down_proj)

        self._initialized = True
    
    def _register_hook(
        self,
        layer_idx: int,
        name: str,
        module: nn.Module
    ):
        """
        Register forward hook with direct statistics collection
        """
        stat = ActStats()
        self._act_stats[layer_idx][name] = stat

        def hook_fn(_, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            stat.update(x)

        handle = self.register_hook(module, hook_fn, "forward")
        self._reorder_hooks[layer_idx][name] = handle

    @torch.inference_mode()
    def apply(self, layer_struct: DecoderStruct, dataloader: CalibDataLoader):
        """
        Apply channel reordering to a single decoder layer

        1. Collect activation statistics via forward pass
        2. Compute channel indices from metrics
        3. Reorder weights
        """
        if not self._initialized:
            raise RuntimeError("ReorderModifier not initialized")
        layer_idx = layer_struct.layer_idx

        # intialize structure
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct

        # if cahced indices not exist, capture activations via hooks
        if layer_idx in self._reorder_hooks:
            # get current hooks
            layer_hooks = self._reorder_hooks[layer_idx]

            # collect activations via forward pass
            dataloader.calibrate(layer_struct, layer_hooks)
            layer_stats = self._act_stats[layer_idx]

            # remove stat hooks
            self.remove_hooks(set(layer_hooks.values()))
            self._reorder_hooks[layer_idx] = {}

        # o proj reordering
        if self.config.reorder_out:
            key = f"layer.{layer_idx}.out"
        
            # get num_heads for gqa
            num_heads = attn.config.num_heads
            num_repeats = attn.config.num_kv_groups

            # compute index
            if key not in self._indices:
                stats = layer_stats.get("o_proj")
                num_channels = attn.o_proj.weight.shape[1]

                idx = compute_index(stats, num_channels, num_heads, num_repeats)
                self._indices[key] = idx.cpu()
            
            # get index
            idx = self._indices[key]
            v_idx = compute_head_index(idx, num_heads, num_repeats)

            # capture signatures for debug
            if layer_idx in self._debug_info:
                sig_o = attn.o_proj.weight[:, idx[0]].sum().item()
                sig_v = attn.v_proj.weight[v_idx[0], :].sum().item()
                self._debug_info[layer_idx]["o_proj"] = (sig_o, sig_v)

            # apply reorder
            reorder(attn.o_proj, idx, attn.v_proj, v_idx)
        
        # down proj reordering
        if self.config.reorder_down:
            key = f"layer.{layer_idx}.down"

            # compute index
            if key not in self._indices:
                stats = layer_stats.get("down_proj")
                num_channels = ffn.down_proj.weight.shape[1]

                idx = compute_index(stats, num_channels)
                self._indices[key] = idx.cpu()

            # get index
            idx = self._indices[key]

            # capture signatures for debug
            if layer_idx in self._debug_info:
                sig_down = ffn.down_proj.weight[:, idx[0]].sum().item()
                sig_up = ffn.up_projs[0].weight[idx[0], :].sum().item()
                self._debug_info[layer_idx]["down_proj"] = (sig_down, sig_up)

            # apply reorder
            reorder(ffn.down_proj, idx, ffn.up_projs)

        # free memory
        del self._act_stats[layer_idx]
    
    def finalize(self):
        """
        Finalize after all layers processed

        1. Save computed indices
        2. Reset states
        """        
        # save indices
        if self.config.path and self._indices:
            torch.save(self._indices, self.config.path)

        # reset state
        self._model_struct = None
        self._indices = {}
        self._reorder_hooks = {}
        self._act_stats = {}
        self._initialized = False
    
    def debug(self, layer_idx: int):
        """
        Log debug information to verify physical weight movement
        """
        if self._model_struct is None or layer_idx < 0:
            return
        
        layer_struct = self._model_struct.layer_structs[layer_idx]
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct
        debug_info = self._debug_info.get(layer_idx, {})

        logger.debug(f"[ReorderModifier] Layer {layer_idx}:")

        # out proj
        if "o_proj" in debug_info:
            trg_o, trg_v = debug_info["o_proj"]
            
            with torch.no_grad():
                cur_o = attn.o_proj.weight[:, 0].sum().item()
                cur_v = attn.v_proj.weight[0, :].sum().item()
            
            logger.debug(f'  o_proj (in): current={cur_o}, target={trg_o}')
            logger.debug(f'  v_proj (out): current={cur_v}, target={trg_v}')
            
        # down_proj
        if "down_proj" in debug_info:
            trg_down, trg_up = debug_info["down_proj"]
            
            with torch.no_grad():
                cur_down = ffn.down_proj.weight[:, 0].sum().item()
                cur_up = ffn.up_projs[0].weight[0, :].sum().item()
            
            logger.debug(f'  down_proj (in): current={cur_down}, target={trg_down}')
            logger.debug(f'  up_proj (out): current={cur_up}, target={trg_up}')