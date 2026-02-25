import os
import torch
import contextlib
import torch.nn as nn

from typing import *
from loguru import logger
from dataclasses import dataclass
from pydantic import PrivateAttr

from compressor.calib import CalibDataLoader
from compressor.nn.struct import LLMStruct, DecoderStruct
from compressor.modifier.base import ModifierConfig, Modifier
from compressor.modifier.rotate.hadamard import (
    get_rotation_matrix, hadamard_in, HadamardMatrix, HadamardInputHook
)
from compressor.modifier.rotate.rotate import (
    rotate_in, rotate_out, fuse_ln_fcs, convert_ln_to_rms
)
from compressor.utils import log_weight

__all__ = ["RotateConfig", "RotateModifier"]

@dataclass
class RotateConfig(ModifierConfig):
    """
    Configuration for QuaRot rotation
    """
    path: str = ""
    random: bool = False

    # modules for additional transforms
    rotate_out: bool = False
    rotate_down: bool = False

class RotateModifier(Modifier):
    """
    QuaRot rotation modifier

    1. Fuses LayerNorm weights into adjacent linear layers
    2. Converts LayerNorm to RMSNorm
    3. Applies Hadamard-based rotation to weight matrices
    """
    config: RotateConfig

    # runtime state
    _model_struct: Optional[LLMStruct] = PrivateAttr(default=None)
    _rotation: Optional[torch.Tensor] = PrivateAttr(default=None)
    _head_rotation: Optional[torch.Tensor] = PrivateAttr(default=None)

    def initialize(self, model_struct: LLMStruct):
        """
        Initalize rotation and process embedding layer
        """
        self._model_struct = model_struct
        config = model_struct.config

        # register hooks for gate function
        if self.config.rotate_down:
            for _, layer_struct in enumerate(model_struct.layer_structs):
                ffn = layer_struct.ffn_struct
                in_channels = ffn.down_proj.in_features
                
                # get haramard components
                rhs, lhs, k = HadamardMatrix.get(
                    in_channels, scale=True,
                    dtype=ffn.down_proj.weight.dtype,
                    device=ffn.down_proj.weight.device,
                )

                # register persistent hooks
                hook = HadamardInputHook(rhs, lhs, k)
                self.register_persistent_hook(
                    ffn.down_proj, hook.as_input_hook(), "forward_pre", with_kwargs=True
                )

        # get rotation matrix
        if self.config.path and os.path.isfile(self.config.path):
            cached = torch.load(self.config.path)
            self._rotation = cached["rotation"]
            self._head_rotation = cached.get("head_rotation")
        else:
            self._rotation = get_rotation_matrix(config.hidden_size, random=self.config.random)
        
        # compute head rotation if needed
        if self.config.rotate_out and self._head_rotation is None:
            head_size = config.hidden_size // config.num_heads
            self._head_rotation = get_rotation_matrix(
                head_size,
                random=self.config.random
            )

        # apply rotation to whole model
        self._rotate()

    @torch.inference_mode()
    def _rotate(self):
        """
        Apply rotation to whole model

        1. Fuse norm scale into adjacent linears & convert LayerNorm → RMSNorm
        2. Apply Hadamard rotation to all weight matrices
        """
        model_struct = self._model_struct
        rotation = self._rotation
        head_rotation = self._head_rotation

        # initialize for first layers 
        prev_ln = [model_struct.embed_tokens]
        prev_out_dim = 1        # (vocab_size, in_channels)

        # fuse norm scales
        for layer_struct in model_struct.layer_structs:
            attn = layer_struct.attn_struct
            ffn = layer_struct.ffn_struct

            # fuse input_layernorm → qkv_projs
            if layer_struct.input_layernorm is not None:
                with move_to_fp32(*attn.qkv_projs, *prev_ln):
                    fuse_ln_fcs(layer_struct.input_layernorm, attn.qkv_projs)
                    rms = convert_ln_to_rms(
                        parent=layer_struct.module,
                        norm_rname=layer_struct.input_layernorm_rname,
                        modules=prev_ln,
                        out_dims=prev_out_dim,
                    )
                    if rms is not None:
                        layer_struct.input_layernorm = rms

            prev_ln = [attn.o_proj]
            prev_out_dim = 0

            # fuse post_attention_layernorm → up_projs
            if layer_struct.post_attention_layernorm is not None:
                with move_to_fp32(*ffn.up_projs, *prev_ln):
                    fuse_ln_fcs(layer_struct.post_attention_layernorm, ffn.up_projs)
                    rms = convert_ln_to_rms(
                        parent=layer_struct.module,
                        norm_rname=layer_struct.post_attention_layernorm_rname,
                        modules=prev_ln,
                        out_dims=prev_out_dim,
                    )
                    if rms is not None:
                        layer_struct.post_attention_layernorm = rms

            prev_ln = [ffn.down_proj]
            prev_out_dim = 0

        # fuse final_norm → lm_head
        with move_to_fp32(model_struct.lm_head, *prev_ln):
            if (
                model_struct.norm is not None
                and model_struct.lm_head is not None
            ):
                fuse_ln_fcs(model_struct.norm, [model_struct.lm_head])

            if model_struct.norm is not None:
                rms = convert_ln_to_rms(
                    parent=model_struct.backbone,
                    norm_rname=model_struct.norm_rname,
                    modules=prev_ln,
                    out_dims=prev_out_dim,
                )
                if rms is not None:
                    model_struct.norm = rms

        # rotate embed_tokens
        rotate_in(model_struct.embed_tokens.weight, rotation)

        # apply offline hadamard rotation
        down_projs = []
        for layer_struct in model_struct.layer_structs:
            attn = layer_struct.attn_struct
            ffn = layer_struct.ffn_struct
            
            with move_to_fp32(*attn.qkv_projs, attn.o_proj):
                # compensate rotation on qkv input
                for proj in attn.qkv_projs:
                    rotate_in(proj.weight, rotation)

                # apply rotation to o_proj output
                rotate_out(attn.o_proj.weight, rotation, attn.o_proj.bias)

                # apply head rotation each value heads
                if head_rotation is not None:
                    rotate_out(attn.v_proj.weight, head_rotation, attn.v_proj.bias)
                    rotate_in(attn.o_proj.weight, head_rotation)

            # compensate rotation on up_projs input
            with move_to_fp32(*ffn.up_projs, ffn.down_proj):
                for proj in ffn.up_projs:
                    rotate_in(proj.weight, rotation)

                # apply rotation to down_proj output
                rotate_out(ffn.down_proj.weight, rotation, ffn.down_proj.bias)

            # append down_proj for hadamard transform
            if self.config.rotate_down:
                down_projs.append(ffn.down_proj)

        # apply online hadamard rotation to down_proj
        if down_projs:
            for module in down_projs:
                with move_to_fp32(module):
                    hadamard_in(module)

        # compensate rotation on lm_head input
        if model_struct.lm_head is not None:
            with move_to_fp32(model_struct.lm_head):
                rotate_in(model_struct.lm_head.weight, rotation)
    
    def apply(self, layer_struct: DecoderStruct, _dataloader: CalibDataLoader):
        """
        Skipped as rotation is applied during the initialization phase
        """
        pass

    def finalize(self):
        """
        Finalize after all layers processed

        1. Save computed rotation
        2. reset states
        """
        # save rotation matrix
        if self.config.path:
            save_dict = {"rotation": self._rotation}
            if self._head_rotation is not None:
                save_dict["head_rotation"] = self._head_rotation
            torch.save(save_dict, self.config.path)

        # reset state
        self._model_struct = None
        self._rotation = None
        self._head_rotation = None

    def debug(self, layer_idx: int):
        """
        Log debug information
        """
        if self._model_struct is None or layer_idx < 0:
            return
        
        layer_struct = self._model_struct.layer_structs[layer_idx]
        attn = layer_struct.attn_struct
        ffn = layer_struct.ffn_struct

        logger.debug(f"[RotateModifier] Layer {layer_idx}:")

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

        # norm
        if layer_struct.input_layernorm is not None:
            log_weight("input_layernorm", layer_struct.input_layernorm.weight)
        if layer_struct.post_attention_layernorm is not None:
            log_weight("post_attention_layernorm", layer_struct.post_attention_layernorm.weight)

@contextlib.contextmanager
def move_to_fp32(*modules: Optional[nn.Module]):
    """
    Temporarily moves Linear modules to CPU in float32 precision
    """
    # filter modules
    modules = [m for m in modules if isinstance(m, nn.Linear)]

    # save current states and convert to fp32
    ori_states = []
    for m in modules:
        ori_states.append((m, m.weight.device, m.weight.dtype))
        m.to(device="cpu", dtype=torch.float32)

    yield

    # return back
    for m, device, dtype in ori_states:
        m.to(device=device, dtype=dtype)