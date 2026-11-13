"""CUDA-synchronised wall-clock timing."""

import time

import torch


def synchronized_time() -> float:
    """``time.perf_counter()`` after waiting for all queued CUDA work (if CUDA is available)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()
