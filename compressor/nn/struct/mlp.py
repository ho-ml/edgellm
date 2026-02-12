import torch.nn as nn

from typing import *
from dataclasses import dataclass, field
if TYPE_CHECKING:
    from compressor.nn.struct.llm import DecoderStruct

__all__ = ["FFNConfig", "FFNStruct"]

@dataclass(kw_only=True)
class FFNConfig:
    """
    Feed-forward module configuration
    """
    hidden_size: int
    intermediate_size: int
    act_type: str = "silu_glu"

    @property
    def is_glu(self):
        return self.act_type.endswith("_glu")
    
@dataclass(kw_only=True)
class FFNStruct:
    """
    Feed-forward module structure for LLM models
    """
    config: Optional[FFNConfig] = field(default=None)
    module: nn.Module = field(repr=False)
    layer_idx: int = 0

    # parent module
    parent: Optional["DecoderStruct"] = field(default=None, repr=False)

    # child modules
    gate_proj: Optional[nn.Linear] = field(default=None, repr=False)
    up_proj: Optional[nn.Linear] = field(default=None, repr=False)
    down_proj: Optional[nn.Linear] = field(default=None, repr=False)

    # module relative names
    gate_proj_rname: str = field(default="gate_proj", repr=False)
    up_proj_rname: str = field(default="up_proj", repr=False)
    down_proj_rname: str = field(default="down_proj", repr=False)

    @property        
    def up_projs(self):
        if self.config and self.config.is_glu:
            return [self.up_proj, self.gate_proj]
        return [self.up_proj]
    
    @property    
    def down_projs(self):
        return [self.down_proj]
    
    @property
    def all_projs(self):
        projs = []
        if self.gate_proj is not None:
            projs.append(self.gate_proj)
        if self.up_proj is not None:
            projs.append(self.up_proj)
        if self.down_proj is not None:
            projs.append(self.down_proj)
        return projs
    
    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        parent: Optional["DecoderStruct"] = None,
        layer_idx: int = 0
    ):
        """
        Construct FFNStruct from an FFN module
        """
        # get child modules
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if gate_proj is None and up_proj is None:
            raise ValueError(f"Unsupported module type: {type(module)}") 
        
        # infer acvitation type
        act_type = infer_act_type(module)

        # determine sizes
        if gate_proj is not None:
            hidden_size = gate_proj.weight.shape[1]
            intermediate_size = gate_proj.weight.shape[0]
        else:
            hidden_size = up_proj.weight.shape[1]
            intermediate_size = up_proj.weight.shape[0]

        # build config
        ffn_config = FFNConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            act_type=act_type
        )

        return cls(
            module=module,
            parent=parent,
            layer_idx=layer_idx,
            config=ffn_config,
            gate_proj=gate_proj,
            up_proj=up_proj,
            down_proj=down_proj,
        )

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Forward pass through the FFN module
        """
        return self.module(*args, **kwargs)

def infer_act_type(module: nn.Module):
    """
    Infer activation type from module
    """
    # default activation
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        raise ValueError(f"FFN module {type(module)} missing act_fn")

    # infer activation
    act_name = type(act_fn).__name__.lower()
    if "silu" in act_name or "swish" in act_name:
        base = "silu"
    elif "gelu" in act_name:
        base = "gelu"
    elif "relu" in act_name:
        base = "relu"
    else:
        base = act_name.replace("activation", "")

    # check if GLU style
    if hasattr(module, "gate_proj"):
        return f"{base}_glu"
    
    return base
