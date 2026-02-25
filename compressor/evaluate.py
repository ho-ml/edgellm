import gc
import json
import torch
import torch.nn as nn

from tqdm import tqdm
from loguru import logger
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import PreTrainedModel, PreTrainedTokenizer
from compressor.utils import get_best_device, get_execution_device
from compressor.data.base import EvalDataset
from compressor.config.evaluator import EvalConfig

__all__ = ["Evaluator"]

@torch.no_grad()
def _compute_ppl(model, dataset, seq_length: int, device):
    """
    Compute perplexity over the given dataset
    """
    nlls = []
    for seq in tqdm(dataset, desc="Computing ppl"):
        # extract input sequence
        batch = seq.unsqueeze(0).to(device)

        # calculate logits
        logits = model(batch).logits

        # first token (not in logits) need to be eliminated
        shift_logits = logits[:, :-1, :].contiguous().float()

        # last token (not in labels) need to be eliminated
        shift_labels = batch[:, 1:].contiguous()

        # calculate total negative log likehood
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        nlls.append(loss.float() * seq_length)

    # calculate the perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (len(dataset) * seq_length))
    return ppl.item()


class Evaluator:
    """
    Pipeline for evaluating a LLM based on EvalConfig
    """
    def __init__(
        self,
        config: EvalConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def run(self, baseline_results: dict | None = None):
        """
        Run evaluation and return results dict
        """
        results = {}

        # device
        device = get_best_device()
        cur_device = get_execution_device(self.model)

        # move model to device
        self.model = self.model.to(device)

        # PPL evaluation
        ppl_results = {}
        for dataset_name in self.config.ppl.datasets:
            # get evaluation dataset from registry
            dataset_cls = EvalDataset.get(dataset_name)
            dataset = dataset_cls(
                tokenizer=self.tokenizer,
                num_samples=self.config.ppl.num_samples,
                seq_length=self.config.ppl.seq_length,
            )

            # compute ppl
            ppl = _compute_ppl(self.model, dataset, self.config.ppl.seq_length, device)
            ppl_results[dataset_name] = ppl
            logger.info(f"PPL [{dataset_name}]: {ppl:.4f}")
        
        results["ppl"] = ppl_results
        self.clear()

        # lm-eval
        if self.config.lm_eval is not None:
            # evaluate
            lm_results = simple_evaluate(
                model=HFLM(pretrained=self.model, tokenizer=self.tokenizer),
                tasks=self.config.lm_eval.tasks,
                num_fewshot=self.config.lm_eval.num_fewshot,
                batch_size=self.config.lm_eval.batch_size,
                limit=self.config.lm_eval.limit,
            )

            # results
            task_results = {}
            for task in self.config.lm_eval.tasks:
                task_data = lm_results["results"].get(task, {})
                acc = task_data.get("acc,none", task_data.get("acc_norm,none"))
                task_results[task] = acc
                logger.info(f"lm-eval [{task}]: {acc}")
            
            results["lm_eval"] = task_results
            self.clear()

        # baseline comparison
        if baseline_results is not None:
            comparison = {}
            for ds, ppl in results["ppl"].items():
                # baseline ppl
                base_ppl = baseline_results.get("ppl", {}).get(ds)
                
                # calculate degradation
                if base_ppl is not None:
                    degradation = (ppl - base_ppl) / base_ppl
                    comparison[f"ppl_{ds}_degradation"] = degradation
                    logger.info(
                        f"PPL [{ds}] baseline={base_ppl:.4f} → compressed={ppl:.4f} "
                        f"(degradation: {degradation:+.2%})"
                    )

            results["comparison"] = comparison
            self.clear()

        # save to json
        if self.config.output_path:
            with open(self.config.output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Eval results saved to {self.config.output_path}")

        # restore model to cpu
        self.model = self.model.to(cur_device)
        self.clear()

    def clear(self):
        """
        Clear memory
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    import sys
    import torch
    import argparse

    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from compressor.config import EvalConfig

    # parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a LLM")
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    parser.add_argument("--model",  required=True, help="Model ID or local path")
    args = parser.parse_args()

    # get config file
    eval_config = EvalConfig.from_yaml(args.config)
    if eval_config is None:
        logger.warning("No evaluation in config.")
        sys.exit(0)

    # get model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # evaluate
    eval = Evaluator(eval_config, model, tokenizer)
    eval.run()
