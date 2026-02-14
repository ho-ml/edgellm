import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from compressor.config import CompressorConfig
from compressor.compress import Compressor
from compressor.evaluate import evaluate

CONFIG_PATH = "examples/configs/qoq.yaml"
MODEL_ID = "meta-llama/Llama-3.2-1B"

# build config
config = CompressorConfig.from_yaml(CONFIG_PATH)

# base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# compress
compressor = Compressor(config, model, tokenizer)
compressed_model = compressor.run()

# evaluate
evaluate(compressed_model, tokenizer)