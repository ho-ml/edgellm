import torch

from tqdm import tqdm
from typing import *
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedTokenizer
from compressor.data import CalibDataset
from compressor.utils import HooksMixin, move_to_device
from compressor.nn.struct.llm import LLMStruct, DecoderStruct
if TYPE_CHECKING:
    from compressor.config.calib import CalibConfig

__all__ = ["CalibDataLoader"]

class CalibDataLoader:
    """
    Layer-wise calibration dataloader with generator
    """
    config: "CalibConfig"
    dataset: CalibDataset
    model_struct: LLMStruct
    onload_device: torch.device
    offload_device: torch.device

    # internal state
    _args: List[torch.Tensor]
    _kwargs: Dict[str, Any]

    def __init__(
        self,
        config: "CalibConfig",
        tokenizer: PreTrainedTokenizer,
        model_struct: LLMStruct,
        onload_device: torch.device,
        offload_device: torch.device
    ):
        """
        Initialize the calibration dataloader
        """
        self.config = config
        self.model_struct = model_struct
        self.onload_device =onload_device
        self.offload_device = offload_device

        # build dataset
        self.dataset = config.build_dataset(tokenizer)

    @torch.inference_mode()
    def initialize(self):
        """
        Prepare first layer's input
        """
        embed = self.model_struct.embed_tokens
        layer = self.model_struct.layer_structs[0].module

        # move to best device
        embed = embed.to(self.onload_device)
        layer = layer.to(self.onload_device)

        # hook function
        captured = {}
        def hook(_mod, _args, kwargs):
            nonlocal captured
            captured = {
                k: move_to_device(v, self.offload_device)
                for k, v in kwargs.items()
            }
            raise StopIteration()

        # register hooks & forward pass
        handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            sample = self.dataset[0].unsqueeze(0).to(self.onload_device)
            self.model_struct(sample)
        except StopIteration:
            pass
        finally:
            handle.remove()
        
        # initialize args & kwargs
        self._args = []
        self._kwargs = captured

        # process kwargs
        if "position_ids" in self._kwargs:
            self._kwargs["position_ids"] = self._kwargs["position_ids"].expand(1, -1)
        if "use_cache" in self._kwargs:
            self._kwargs["use_cache"] = False
        if "past_key_values" in self._kwargs:
            self._kwargs["past_key_values"] = None
        
        # pass embedding layers
        for sample in self.dataset:
            input_ids = sample.unsqueeze(0).to(self.onload_device)

            # embed
            output = self.model_struct.embed_tokens(input_ids)
            self._args.append(output.cpu())

        # move to original device
        embed = embed.to(self.offload_device)
        layer = layer.to(self.offload_device)

    def iter_samples(self):
        """
        Iterate over samples for the current layer
        """
        for i in range(0, len(self._args)):
            # hidden states
            args = self._args[i].to(self.onload_device)
                
            yield args

    @torch.inference_mode()
    def calibrate(
        self, layer_struct: DecoderStruct, layer_hooks: Dict[str, RemovableHandle]
    ):
        """
        Collect actviations from given layer
        """
        kwargs = {
            k: move_to_device(v, self.onload_device) for k, v in self._kwargs.items()
        }

        with HooksMixin.disable_hooks(keep=set(layer_hooks.values())):
            for args in tqdm(self.iter_samples(), total=len(self._args), desc="Calibration"):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                # forward
                _ = layer_struct(args, **kwargs)
        
        del kwargs
        
    @torch.inference_mode()
    def propagate(self, layer_struct: DecoderStruct):
        """
        Update input args & kwargs of layer
        """
        kwargs = {
            k: move_to_device(v, self.onload_device) for k, v in self._kwargs.items()
        }

        with HooksMixin.disable_hooks():
            for i, args in tqdm(enumerate(self.iter_samples()), total=len(self._args), desc="Propagation"):
                # forward pass
                output = layer_struct(args, **kwargs)
                if isinstance(output, tuple):
                    output = output[0]

                # update
                self._args[i] = output.cpu()

        del kwargs
