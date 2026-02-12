import os
import torch
import psutil

from loguru import logger
from compressor.utils.stats import ActStats

__all__ = ["log_weight", "log_activation", "log_memory"]

def log_weight(name: str, weight: torch.Tensor):
    """
    Log weight statistics: shape, mean, std, absmaxs
    """
    w = weight.data.float()

    info = {
        "shape": tuple(w.shape),
        "mean": w.mean().item(),
        "std": w.std().item(),
        "absmax": w.abs().max().item()
    }

    logger.debug(
        f"  {name}: shape={info['shape']}, "
        f"mean={info['mean']:.6f}, std={info['std']:.6f}, "
        f"absmax={info['absmax']:.6f}"
    )

    return info

def log_activation(stats: ActStats, indices: list = None):
    """
    Log activation statistics: absmin, absmax, specific indices 
    """
    max_vals = stats.absmax.float()

    info = {
        "min": max_vals.min().item(),
        "max": max_vals.max().item()
    }

    for idx in indices:
        info[f"idx{idx}"] = max_vals[idx].item()

    return info

def log_memory():
    """
    Log memory usage: cpu, gpu
    """
    # ram
    cur_ram = psutil.virtual_memory().percent

    # vram
    if torch.cuda.is_available():
        max_vram = torch.cuda.max_memory_allocated() / 1024**3
        cur_vram = torch.cuda.memory_allocated() / 1024**3

    logger.debug(
        f"  RAM: {cur_ram}%, VRAM: {cur_vram:.3f}GB, Peak VRAM: {max_vram:.3f}GB"
    )