import torch
import torch.nn as nn

from typing import *
from dataclasses import dataclass, field
from transformers import PreTrainedModel, PretrainedConfig
from compressor.nn.struct.attn import AttentionStruct
from compressor.nn.struct.mlp import FFNStruct
from compressor.nn.patch import patch_rope, patch_norm

__all__ = [
    "LLMConfig",
    "DecoderStruct",
    "LLMStruct",
]

@dataclass(kw_only=True)
class LLMConfig:
    """
    LLM configuration
    """
    # input / output activation
    hidden_size: int

    # attention
    num_heads: int
    kv_heads: int

    # ffn
    intermediate_size: int
    act_type: str = "silu_glu"

    # model config
    num_hidden_layers: int
    vocab_size: int
    tie_word_embeddings: bool = False

    # position embedding
    with_rope: bool = True

    @classmethod
    def from_pretrained(cls, config: PretrainedConfig):
        """
        Create config from huggingface config
        """
        act_type = getattr(config, "hidden_act", "silu")
        if hasattr(config, "intermediate_size"):
            act_type = f"{act_type}_glu"

        # detect rope
        with_rope = (
            hasattr(config, "rope_theta") or
            hasattr(config, "rope_scaling") or
            getattr(config, "position_embedding_type", "rope") == "rope"
        )

        return cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            vocab_size=config.vocab_size,
            act_type=act_type.lower(),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            with_rope=with_rope,
        )

@dataclass(kw_only=True)
class DecoderStruct:
    """
    Single decoder layer struct
    """
    module: nn.Module = field(repr=False)
    layer_idx: int = 0

    # parent module
    parent: Optional["LLMStruct"] = field(default=None, repr=False)

    # layer norms
    input_layernorm: Optional[nn.Module] = field(default=None, repr=False)
    post_attention_layernorm: Optional[nn.Module] = field(default=None, repr=False)
    pre_feedforward_layernorm: Optional[nn.Module] = field(default=None, repr=False)
    post_feedforward_layernorm: Optional[nn.Module] = field(default=None, repr=False)

    # child modules
    attn: Optional[nn.Module] = field(default=None, repr=False)
    mlp: Optional[nn.Module] = field(default=None, repr=False)
    attn_struct: Optional[AttentionStruct] = field(default=None, repr=False)
    ffn_struct: Optional[FFNStruct] = field(default=None, repr=False)

    # module relative names
    attn_rname: str = field(default="self_attn", repr=False)
    mlp_rname: str = field(default="mlp", repr=False)
    input_layernorm_rname: str = field(default="input_layernorm", repr=False)
    post_attention_layernorm_rname: str = field(default="post_attention_layernorm", repr=False)

    @property
    def name(self):
        """
        Get the full name of this layer
        """
        if self.parent:
            return f"{self.parent.layers_name}.{self.layer_idx}"
        return f"layers.{self.layer_idx}"

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        parent: Optional["LLMStruct"] = None,
        layer_idx: int = 0,
        with_rope: bool = True
    ):
        """
        Construct DecoderStruct from a decoder layer
        """
        # get layer norms
        input_layernorm = getattr(module, "input_layernorm", None)
        post_attention_layernorm = getattr(module, "post_attention_layernorm", None)
        pre_feedforward_layernorm = getattr(module, "pre_feedforward_layernorm", None)
        post_feedforward_layernorm = getattr(module, "post_feedforward_layernorm", None)

        # get attention module
        attn = getattr(module, "self_attn", None)
        attn_rname = "self_attn"

        # get mlp module
        mlp = getattr(module, "mlp", None)
        mlp_rname = "mlp"

        # build struct without child structs
        struct = cls(
            module=module,
            parent=parent,
            layer_idx=layer_idx,
            input_layernorm=input_layernorm,
            post_attention_layernorm=post_attention_layernorm,
            pre_feedforward_layernorm=pre_feedforward_layernorm,
            post_feedforward_layernorm=post_feedforward_layernorm,
            attn=attn,
            mlp=mlp,
            attn_rname=attn_rname,
            mlp_rname=mlp_rname,
        )

        # parse child modules
        if attn is not None:
            struct.attn_struct = AttentionStruct.from_module(attn, parent=struct, layer_idx=layer_idx, with_rope=with_rope)
        if mlp is not None:
            struct.ffn_struct = FFNStruct.from_module(mlp, parent=struct, layer_idx=layer_idx)

        return struct

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Forward pass through the decoder layer
        """
        return self.module(*args, **kwargs)

@dataclass(kw_only=True)
class LLMStruct:
    """
    LLM model structure (CausalLM)

    Supports: LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM
    """
    config: Optional[LLMConfig] = field(default=None)
    model: PreTrainedModel = field(repr=False)

    # backbone (model.model)
    backbone: Optional[nn.Module] = field(default=None, repr=False)
    backbone_rname: str = field(default="model", repr=False)

    # lm_head
    lm_head: Optional[nn.Linear] = field(default=None, repr=False)
    lm_head_rname: str = field(default="lm_head", repr=False)

    # backbone child modules
    embed_tokens: Optional[nn.Embedding] = field(default=None, repr=False)
    layers: Optional[nn.ModuleList] = field(default=None, repr=False)
    norm: Optional[nn.Module] = field(default=None, repr=False)
    rotary_emb: Optional[nn.Module] = field(default=None, repr=False)

    # parsed layer structs
    layer_structs: List[DecoderStruct] = field(default_factory=list, repr=False)

    # relative names
    embed_tokens_rname: str = field(default="embed_tokens", repr=False)
    layers_rname: str = field(default="layers", repr=False)
    norm_rname: str = field(default="norm", repr=False)

    @property
    def layers_name(self):
        """
        Get the full name of the layers module list
        """
        return f"{self.backbone_rname}.{self.layers_rname}"

    @property
    def num_layers(self):
        return len(self.layers) if self.layers else 0

    def embed(self, input_ids: torch.Tensor):
        """
        Get embeddings for input tokens
        """
        if self.embed_tokens is None:
            raise ValueError("embed_tokens not found")
        return self.embed_tokens(input_ids)

    @classmethod
    def from_model(cls, model: PreTrainedModel):
        """
        Construct LLMStruct from a HuggingFace CausalLM model
        """
        config = LLMConfig.from_pretrained(model.config)

        # get backbone (model.model)
        backbone = getattr(model, "model", None)
        backbone_rname = "model"
        if backbone is None:
            raise ValueError(f"Could not find backbone in model {type(model)}")

        # get lm_head
        lm_head = getattr(model, "lm_head", None)

        # get backbone child modules
        embed_tokens = getattr(backbone, "embed_tokens", None)
        layers = getattr(backbone, "layers", None)
        norm = getattr(backbone, "norm", None)
        rotary_emb = getattr(backbone, "rotary_emb", None)

        # build struct
        struct = cls(
            config=config,
            model=model,
            backbone=backbone,
            backbone_rname=backbone_rname,
            lm_head=lm_head,
            embed_tokens=embed_tokens,
            layers=layers,
            norm=norm,
            rotary_emb=rotary_emb,
        )

        # parse layer structs
        if layers is not None:
            struct.layer_structs = [
                DecoderStruct.from_module(layer, parent=struct, layer_idx=idx, with_rope=config.with_rope)
                for idx, layer in enumerate(layers)
            ]

        return struct

    def patch(self):
        """
        Apply all model patches
        """
        # patch rope
        if self.config and self.config.with_rope:
            patch_rope(self.model)
        
        # patch norm for gemma
        patch_norm(self.model)
        
        return self

    def iter_layers(self):
        """
        Iterate over all layer index & struct pairs
        """
        for idx, layer_struct in enumerate(self.layer_structs):
            yield idx, layer_struct

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Forward pass through the model
        """
        return self.model(*args, **kwargs)
