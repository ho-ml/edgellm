import torch
import random

from typing import *
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from compressor.utils.registry import RegistryMixin

__all__ = ["CalibDataset"]

class CalibDataset(RegistryMixin, Dataset):
    """
    Base dataset class for calibration
    """
    # raw data config
    data_name_or_path: str
    data_config: str | None
    split: str
    text_key: str

    # preprocessed data for calibration
    data: List[torch.Tensor]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_samples: int,
        seq_length: int,
        max_length: int = -1,
        min_length: int = -1,
        seed: int = 42
    ):
        # sanity check
        assert num_samples > 0, "num_samples should be larger than 0"
        assert seq_length > 0, "seq_length should be larger than 0"
        assert tokenizer is not None, "tokenizer is required"
        
        # initialize
        random.seed(seed)
        num_tokens = num_samples * seq_length

        # load raw dataset
        dataset = self.get_dataset(seed)

        # preprocess for calibration
        seqs, toks = [], 0
        for sample in dataset:
            text = sample[self.text_key]
            
            # tokenize
            line = tokenizer.encode(text.strip())

            # skip if length not satisfy the conditions
            length = len(line)
            if length == 0:
                continue
            if min_length > 0 and length < min_length:
                continue
            if max_length > 0 and length > max_length:
                continue
            
            # random crop
            seq = torch.tensor(line)
            if length > seq_length:
                tok = random.randint(0, length - seq_length)
                seq = seq[tok: tok + seq_length]
            
            # update
            seqs.append(seq)
            toks += seq.numel()

            # stop if collect enough samples
            if len(seqs) >= num_samples and toks >= num_tokens:
                break

        # concat & split
        seqs = torch.cat(seqs).split(seq_length)
        if toks > num_tokens:
            seqs = seqs[:-1]
        seqs = seqs[:num_samples]
        self.data = seqs

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_dataset(self, seed: int):
        """
        Return raw dataset from the given data
        """
        dataset = load_dataset(
            self.data_name_or_path,
            self.data_config,
            split=self.split,
            streaming=True
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        
        return dataset