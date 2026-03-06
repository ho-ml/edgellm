from typing import *
from dataclasses import dataclass

__all__ = ["PackConfig"]

@dataclass
class PackConfig:
    """
    Configuration for packing compressed models
    """
    enabled: bool = False
    output_dir: str = ""
    max_shard_size: float = 4.0
    mma: bool = False
    dtype: str = "float16"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Create config from dictionary
        """
        return cls(**
            {
                k: v
                for k, v in config_dict.items()
                if k in cls.__dataclass_fields__
            }
        )
