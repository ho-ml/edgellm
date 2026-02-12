import torch

from typing import *
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from compressor.data import CalibDataset
from compressor.nn import LLMStruct
from compressor.calib.loader import CalibDataLoader

__all__ = ["CalibConfig"]

@dataclass
class CalibConfig:
    """
    Configuration for calibration dataset & dataloader
    """
    data: str
    num_samples: int
    seq_length: int
    min_length: int = 0
    max_length: int = 0
    seed: int = 42

    def __post_init__(self):
        self.min_length = max(0, self.min_length)
        self.max_length = max(0, self.max_length)

    def build_dataset(self, tokenizer: PreTrainedTokenizer):
        """
        Build calibration dataset from registry
        """
        dataset_cls = CalibDataset.get(self.data)
        dataset = dataset_cls(
            tokenizer=tokenizer,
            num_samples=self.num_samples,
            seq_length=self.seq_length,
            max_length=self.max_length,
            min_length=self.min_length,
            seed=self.seed
        )

        return dataset
    
    def build_dataloader(
        self, 
        model_struct: LLMStruct,
        tokenizer: PreTrainedTokenizer, 
        onload_device: torch.device,
        offload_device: torch.device,
    ):
        """
        Build calibration dataloader
        """
        dataloader = CalibDataLoader(
            config=self, 
            model_struct=model_struct, 
            tokenizer=tokenizer, 
            onload_device=onload_device,
            offload_device=offload_device
        )

        return dataloader
