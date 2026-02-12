import torch
import torch.nn as nn

from tqdm import tqdm
from loguru import logger
from datasets import load_dataset
from compressor.utils import get_best_device

__all__ = ["evaluate"]

NUM_SAMPLE = 40
MAX_LENGTH = 2048

def evaluate(model, tokenizer):
    # get device
    device = get_best_device()

    # move to gpu
    model = model.to(device)

    # load and tokenize the raw dataset
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer('\n\n'.join(testenc['text']), return_tensors='pt')
    testenc = testenc.input_ids.to(device)
    
    nlls = []
    for i in tqdm(range(NUM_SAMPLE), desc="evaluating..."):
        # extract input sequence, chunk for 2048 (max context length)
        batch = testenc[:, i * MAX_LENGTH: (i + 1) *  MAX_LENGTH].to(device)

        # calculate logits
        with torch.no_grad():
            logits = model(batch).logits

        # last token (not in labels) need to be eliminated
        shift_logits = logits[:, :-1, :].contiguous().float()

        # first token (not in logits) need to be eliminated
        shift_labels = testenc[:, (i * MAX_LENGTH):((i + 1) *  MAX_LENGTH)][:, 1:]

        # calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # calculate total negative log likehood
        negative_log_likehood = loss.float() * MAX_LENGTH
        nlls.append(negative_log_likehood)
    
    # calculate the perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (NUM_SAMPLE * MAX_LENGTH))
    logger.info(f"perplexity: {ppl}")

    return ppl
    