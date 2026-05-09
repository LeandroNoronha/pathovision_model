"""Device and mixed precision utilities."""

import logging
from contextlib import contextmanager
from typing import Generator

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the best available device.

    Returns:
        torch.device for CUDA GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Using GPU: %s (%.1f GB)", gpu_name, gpu_mem)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU. Training will be very slow.")

    return device


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """Get the appropriate AMP dtype for the device.

    Args:
        device: The compute device.

    Returns:
        torch.float16 for CUDA, torch.bfloat16 if supported, else float32.
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def print_gpu_info() -> None:
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("No CUDA GPU available.")
        return

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {total_mem:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")

    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")


@contextmanager
def gpu_memory_tracker(label: str = "") -> Generator[None, None, None]:
    """Context manager to track GPU memory usage.

    Args:
        label: Optional label for the memory report.
    """
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / (1024**2)

    yield

    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)

    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sGPU Memory: %.1f MB -> %.1f MB (peak: %.1f MB)",
        prefix, mem_before, mem_after, peak,
    )
