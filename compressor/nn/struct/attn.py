import torch.nn as nn

from typing import *
from dataclasses import dataclass, field
if TYPE_CHECKING:
    from compressor.nn.struct.llm import DecoderStruct

__all__ = ["AttentionConfig", "AttentionStruct"]

@dataclass(kw_only=True)
class AttentionConfig:
    """
    Attention module configuration
    """
    hidden_size: int
    inner_size: int
    num_heads: int
    kv_heads: int
    with_rope: bool = True

    @property
    def head_size(self):
        return self.inner_size // self.num_heads
    
    @property
    def num_kv_groups(self):
        return self.num_heads // self.kv_heads
    
    @property
    def query_size(self):
        return self.inner_size
    
    @property
    def kv_size(self):
        return self.head_size * self.kv_heads
    
@dataclass(kw_only=True)
class AttentionStruct:
    """
    Attention module structure for LLM models
    """
    config: Optional[AttentionConfig] = field(default=None)
    module: nn.Module = field(repr=False)
    layer_idx: int = 0

    # parent module
    parent: Optional["DecoderStruct"] = field(default=None, repr=False)

    # child modules
    q_proj: Optional[nn.Linear] = field(default=None, repr=False)
    k_proj: Optional[nn.Linear] = field(default=None, repr=False)
    v_proj: Optional[nn.Linear] = field(default=None, repr=False)
    o_proj: Optional[nn.Linear] = field(default=None, repr=False)

    # rotary embedding modules
    q_rotary_emb: Optional[nn.Module] = field(default=None, repr=False)
    k_rotary_emb: Optional[nn.Module] = field(default=None, repr=False)
    
    # module relative names
    q_proj_rname: str = field(default="q_proj", repr=False)
    k_proj_rname: str = field(default="k_proj", repr=False)
    v_proj_rname: str = field(default="v_proj", repr=False)
    o_proj_rname: str = field(default="o_proj", repr=False)

    # kwargs for attention forward
    _kwargs: Tuple = field(
        default=(
            "attention_mask",
            "position_ids",
            "past_key_values",
            "output_attentions",
            "use_cache",
            "position_embeddings",
            "cache_position",
        ),
        repr=False
    )

    @property
    def qkv_projs(self):
        return [self.q_proj, self.k_proj, self.v_proj]

    @property
    def all_projs(self):
        return [self.q_proj, self.k_proj, self.v_proj, self.o_proj]
    
    def filter_kwargs(self, kwargs: Dict):
        """
        Return layer kwargs for attention module forward
        """
        return {k: v for k, v in kwargs.items() if k in self._kwargs}
    
    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        parent: Optional["DecoderStruct"] = None,
        layer_idx: int = 0,
        with_rope: bool = True
    ):
        """
        Construct AttentionStruct from an attention module
        """
        # get module config
        if hasattr(module, "config"):
            config = module.config
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")
        num_heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", num_heads)

        # get child modules
        q_proj = getattr(module, "q_proj", None)
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        o_proj = getattr(module, "o_proj", None)

        # get rotary embedding modules
        q_rotary_emb = getattr(module, "q_rotary_emb", None)
        k_rotary_emb = getattr(module, "k_rotary_emb", None)

        # raise error if q_proj & o_proj are None
        if q_proj is None:
            raise ValueError(f"Attention module {type(module)} missing q_proj")
        if o_proj is None:
            raise ValueError(f"Attention module {type(module)} missing o_proj")

        # build config
        attn_config = AttentionConfig(
            hidden_size=q_proj.weight.shape[1],
            inner_size=q_proj.weight.shape[0],
            num_heads=num_heads,
            kv_heads=kv_heads,
            with_rope=with_rope,
        )

        return cls(
            module=module,
            parent=parent,
            layer_idx=layer_idx,
            config=attn_config,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            q_rotary_emb=q_rotary_emb,
            k_rotary_emb=k_rotary_emb,
        )
    
    def __call__(self, *args: Any, **kwargs: Any):
        """
        Forward pass through the attention module
        """
        return self.module(*args, **kwargs)