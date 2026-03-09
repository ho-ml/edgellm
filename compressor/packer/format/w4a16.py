import torch
import numpy as np

from typing import *
from compressor.utils import quantize
from compressor.config.quant import QuantArgs
from compressor.packer.format.base import FormatHandler

__all__ = ["W4A16Format"]

TILE_SIZE = 16

@FormatHandler.register("w4a16")
class W4A16Format(FormatHandler):
    """
    Marlin-compatible W4A16 format handler
    """
    def __init__(self, args: QuantArgs):
        super().__init__(args)

    def convert_linear(
        self,
        weight: torch.Tensor,
        qparams: Dict[str, torch.Tensor],
        qparams_key: str,
        dtype: torch.dtype,
    ):
        scale = qparams[f"{qparams_key}.scale"]
        zero = qparams[f"{qparams_key}.zero"]

        # quantize weight to int8
        q = quantize(
            weight.float(),
            scale.float(),
            zero.float(),
            self.args,
            dtype=torch.int8
        )

        # sanity check
        K, N = weight.shape
        assert K % TILE_SIZE == 0
        assert N % TILE_SIZE == 0

        # resolve group_size
        group_size = self._get_group_size(K)

        # conver to sint4 to uint4
        q = (q.int() + 8).to(torch.int32)

        # pack weight to marlin format
        qweight = self.pack_weight(q, K, N)

        # apply permutation to scale & zero
        scales, zeros = self.pack_params(scale, zero, K, N, group_size)

        result = {
            "qweight": qweight,
            "scales": scales.to(dtype),
        }
        
        if not self.args.symmetric:
            result["zeros"] = zeros

        return result
    
    def pack_weight(
        self,
        weight: torch.Tensor,
        nrows: int,
        ncols: int
    ):
        """
        Pack weight according to MARLIN format

        Tiling → Permutation → Pack
        """
        # tiling
        qweight = weight.reshape(
            nrows // TILE_SIZE, TILE_SIZE, ncols // TILE_SIZE, TILE_SIZE            # (K // 16, 16, N // 16, 16)
        )
        qweight = qweight.permute(0, 2, 1, 3)                                       # (K // 16, N // 16, 16, 16)
        qweight = qweight.reshape(nrows // TILE_SIZE, ncols * TILE_SIZE)            # (K // 16, N * 16)

        # permutation
        perm = _get_weight_perms()
        qweight = qweight.reshape(-1, perm.numel())[:, perm].reshape(qweight.shape)

        # pack uint4 to int32
        q_packed = _pack_int4(qweight)
        
        return q_packed

    def pack_params(
        self,
        scale: torch.Tensor,
        zero: torch.Tensor,
        nrows: int,
        ncols: int,
        group_size: int
    ):
        """
        Pack scale & zero according to MARLIN format
        """
        # get permutation
        perm_group, perm_channel = _get_params_perms()

        # apply to scale
        if group_size < nrows and group_size != -1:
            # per-group quantization
            scale = scale.reshape(-1, len(perm_group))[:, perm_group]
        else: 
            # per-channel quantization
            scale = scale.reshape(-1, len(perm_channel))[:, perm_channel]
        scale = scale.reshape(-1, ncols).contiguous()
        
        # apply to zero if assymetric
        if not self.args.symmetric:
            # permutation
            zero = zero.reshape(-1, len(perm_group))[:, perm_group]

            # apply interleaving
            indices = np.array([0, 2, 4, 6, 1, 3, 5, 7])
            zero = zero.reshape(-1, len(indices))[:, indices].ravel()
            zero = zero.reshape(-1, ncols).contiguous()
            
            # pack
            zero = _pack_int4(zero)

        return scale, zero
        
def _get_weight_perms():
    """
    Weight permutation indices

    1. permute weight for optimizing warp-level access (`ldmatrix` can access 4bytes at once)
    2. apply interleaving to weight for optimizing dequantization
    """
    perm = []
    
    # a warp has 32 threads
    for i in range(32):
        perm1 = []
        
        # calculate indices for a current thread
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4), 2 * (i % 4) + 1,
                2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1
            ]:
                # (row, col + block * 8)
                perm1.append(16 * row + col + 8 * block)

        # unrolling for 16 x 64 tile
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
  
    # apply interleaving
    perm = np.array(perm)
    indices = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape(-1, len(indices))[:, indices].ravel()
    
    return torch.from_numpy(perm)
    
def _get_params_perms():
    """
    Scale & zero permutation indices
    """
    # per-group
    perm_group = []
    for i in range(8):
        perm_group.extend([i + 8 * j for j in range(8)])
    
    # per-channel
    perm_channel = []
    for i in range(4):
        perm_channel.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])

    return perm_group, perm_channel

def _pack_int4(x: torch.Tensor):
    """
    Pack uint4 to int32
    """
    # sanity check
    K, N = x.shape
    assert N % 8 == 0

    # initialize for packing
    ori_device = x.device
    x_numpy = x.cpu().numpy().astype(np.uint32)
    x_packed = np.zeros((K, N // 8), dtype=np.uint32)

    # pack uint4 to int32
    for i in range(8):
        x_packed |= x_numpy[:, i::8] << (4 * i)
    
    return torch.from_numpy(x_packed.astype(np.int32)).to(ori_device)