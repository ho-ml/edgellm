import os
import yaml

from typing import *
from dataclasses import dataclass, field

__all__ = ["EvalConfig"]

DEFAULT_LM_EVAL_TASKS = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge"]
DEFAULT_PPL_DATASETS = ["wikitext"]


@dataclass
class PPLEvalConfig:
    """
    Configuration for calculating PPL
    """
    datasets: list = field(default_factory=lambda: list(DEFAULT_PPL_DATASETS))
    num_samples: int = 128
    seq_length: int = 2048


@dataclass
class LmEvalConfig:
    """
    Benchmark tasks from `lm_eval` library
    """
    tasks: list = field(default_factory=lambda: list(DEFAULT_LM_EVAL_TASKS))
    num_fewshot: int = 0
    batch_size: int = 4
    limit: Optional[int] = None


@dataclass
class EvalConfig:
    ppl: PPLEvalConfig = field(default_factory=PPLEvalConfig)
    lm_eval: Optional[LmEvalConfig] = None
    compare_baseline: bool = False
    output_path: Optional[str] = None

    def __post_init__(self):
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str):
        """
        Parse only the eval section from yaml
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        eval_dict = config.get("eval", None)
        if eval_dict is None:
            return None

        ppl_dict = eval_dict.get("ppl", None)
        lm_eval_dict = eval_dict.get("lm_eval", None)

        return cls(
            ppl=PPLEvalConfig(**ppl_dict) if ppl_dict is not None else PPLEvalConfig(),
            lm_eval=LmEvalConfig(**lm_eval_dict) if lm_eval_dict is not None else None,
            compare_baseline=eval_dict.get("compare_baseline", False),
            output_path=eval_dict.get("output_path", None),
        )
