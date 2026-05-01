"""Computational efficiency metrics: FLOPs, inference time, model size."""

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary with 'total', 'trainable', 'frozen' parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def measure_model_size(model: nn.Module) -> float:
    """Measure model size in MB (state dict).

    Args:
        model: PyTorch model.

    Returns:
        Model size in megabytes.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_inference_time(
    model: nn.Module,
    input_size: tuple[int, int] = (500, 500),
    device: torch.device | None = None,
    num_warmup: int = 10,
    num_runs: int = 50,
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure inference time per image.

    Args:
        model: PyTorch model.
        input_size: (H, W) input size.
        device: Compute device.
        num_warmup: Warmup iterations (not timed).
        num_runs: Timed iterations.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with 'mean_ms', 'std_ms', 'median_ms', 'fps'.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy = torch.randn(batch_size, 3, *input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000 / batch_size)  # ms per image

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def estimate_flops(
    model: nn.Module,
    input_size: tuple[int, int] = (500, 500),
) -> dict[str, Any]:
    """Estimate FLOPs using fvcore or ptflops if available.

    Args:
        model: PyTorch model.
        input_size: (H, W) input size.

    Returns:
        Dictionary with 'flops' (in GFLOPs) and 'method' used.
    """
    dummy = torch.randn(1, 3, *input_size)
    device = next(model.parameters()).device

    # Try fvcore first
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = dummy.to(device)
        flops_analysis = FlopCountAnalysis(model, dummy)
        flops = flops_analysis.total() / 1e9
        return {"gflops": float(flops), "method": "fvcore"}
    except ImportError:
        pass

    # Try ptflops
    try:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(
            model, (3, *input_size),
            as_strings=False, print_per_layer_stat=False,
        )
        return {"gflops": float(flops / 1e9), "method": "ptflops"}
    except ImportError:
        pass

    # Fallback: rough estimate from parameter count
    total_params = sum(p.numel() for p in model.parameters())
    estimated_flops = total_params * 2 * input_size[0] * input_size[1] / (224 * 224)
    return {
        "gflops": float(estimated_flops / 1e9),
        "method": "estimated (install fvcore or ptflops for accuracy)",
    }


def benchmark_model(
    model: nn.Module,
    input_size: tuple[int, int] = (500, 500),
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run full efficiency benchmark on a model.

    Args:
        model: PyTorch model.
        input_size: (H, W) input size.
        device: Compute device.

    Returns:
        Dictionary with all efficiency metrics.
    """
    params = count_parameters(model)
    size_mb = measure_model_size(model)
    timing = measure_inference_time(model, input_size, device)
    flops = estimate_flops(model, input_size)

    result = {
        "params_total_M": params["total"] / 1e6,
        "params_trainable_M": params["trainable"] / 1e6,
        "model_size_MB": size_mb,
        "inference_mean_ms": timing["mean_ms"],
        "inference_std_ms": timing["std_ms"],
        "fps": timing["fps"],
        "gflops": flops["gflops"],
        "flops_method": flops["method"],
    }

    logger.info(
        "Benchmark: %.2fM params, %.1f MB, %.1f ms/img (%.1f FPS), %.2f GFLOPs",
        result["params_total_M"], result["model_size_MB"],
        result["inference_mean_ms"], result["fps"], result["gflops"],
    )

    return result
