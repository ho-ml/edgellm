from datasets import load_dataset
from compressor.data.base import CalibDataset

__all__ = ["UltraChatDataset"]

DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n' + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
)

@CalibDataset.register("ultrachat")
class UltraChatDataset(CalibDataset):
    """
    HuggingFace UltraChat 200k dataset
    """
    data_name_or_path = "HuggingFaceH4/ultrachat_200k"
    data_config = None
    split = "train_sft"
    text_key = "text"

    # TODO: fix error

    def __init__(
        self, 
        tokenizer, 
        num_samples, 
        seq_length, 
        max_length=-1, 
        min_length=-1, 
        seed=42
    ):
        # set chat template if not exists
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        super().__init__(
            tokenizer=tokenizer,
            num_samples=num_samples,
            seq_length=seq_length,
            max_length=max_length,
            min_length=min_length,
            seed=seed
        )

    def get_dataset(self, seed: int):
        """
        Load and preprocess ultrachat dataset
        """
        # load dataset
        dataset = load_dataset(
            self.data_name_or_path,
            self.data_config,
            split=self.split,
            streaming=True
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)

        def apply_chat_template(sample):
            messages = sample["messages"]
            
            # add system message if not exists
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})

            # convert messages to text using map
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            return {"text": text}

        # store tokenizer reference for map function
        self._tokenizer = self.__dict__.get("_init_tokenizer")
        dataset = dataset.map(apply_chat_template)

        return dataset
