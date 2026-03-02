import torch
from compressor.config.quant import QuantConfig

__all__ = ["infer_format", "pack_int4", "unpack_int4", "apply_mma"]

def infer_format(quant_config: QuantConfig):
    """
    Infer format from quantization config
    """
    wbits = quant_config.weight.bits
    has_input = quant_config.input is not None

    if wbits == 4:
        if has_input: return "w4a8"
        else: return "w4a16"
    
    elif wbits == 8:
        if has_input: return "w8a8"

    raise ValueError("Unsupported format")

def pack_int4(data: torch.Tensor):
    """
    Pack two int4 values into one int8 
    """
    # sanity check
    assert data.shape[-1] % 2 == 0

    # seperate even & odd indicies of data
    low = data[..., 0::2].to(torch.int8) & 0x0F
    high = data[..., 1::2].to(torch.int8) & 0x0F

    # shift odd data to high nibble
    high = high << 4

    # merge
    return (low | high).to(torch.int8)

def unpack_int4(packed: torch.Tensor, dtype: torch.dtype = torch.int8):
    """
    Unpack int8 back to two int4 values
    """
    # seperate low & high nibbles
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # sign extension
    low = torch.where(low > 7, low - 16, low).to(dtype)
    high = torch.where(high > 7, high - 16, high).to(dtype)

    # interleave back
    result = torch.stack([low, high], dim=-1).flatten(start_dim=-2)
    
    return result

def apply_mma(
    tensor: torch.Tensor, format_type: str, tensor_type: str
):
    """
    Rearrange tensor for MMA tile layout
    """
    # TODO: implement after CUDA kernel implementation
    return tensor
    