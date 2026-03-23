"""Unified device management for cross-platform compatibility (CUDA / MPS / CPU)."""

import torch


def resolve_device(config_device="auto"):
    """Resolve the compute device from config value.

    "auto" probes in order: cuda → mps → cpu.
    An explicit name is validated and returned as-is.
    """
    if config_device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Explicit device — quick sanity check
    if config_device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable, falling back to CPU")
        return "cpu"
    if config_device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but unavailable, falling back to CPU")
        return "cpu"
    return config_device


def resolve_dtype(config_dtype="auto", device="cpu"):
    """Decide torch dtype based on config and device.

    - CUDA: float16 saves VRAM (typically 4-8 GB).
    - MPS (Apple Silicon): float32 is more stable; unified memory is large enough.
    - CPU: float32 (float16 gives no speed benefit on CPU).
    """
    if config_dtype == "auto":
        if device == "cuda":
            return torch.float16
        return torch.float32

    return torch.float16 if config_dtype == "float16" else torch.float32


def empty_cache(device):
    """Release accelerator memory cache."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def is_oom_error(exc):
    """Return True if *exc* is an out-of-memory error on any backend."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def generator_device(device):
    """Return the device string suitable for ``torch.Generator``.

    MPS does not support Generator on-device; use CPU instead.
    """
    if device == "mps":
        return "cpu"
    return device
