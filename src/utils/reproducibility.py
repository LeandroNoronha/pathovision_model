"""Reproducibility utilities for deterministic training."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Random seed set to %d (deterministic mode)", seed)


def set_seed_fast(seed: int = 42) -> None:
    """Set random seeds but allow non-deterministic cuDNN for speed.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Allow cuDNN autotuner for speed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info("Random seed set to %d (fast mode, non-deterministic cuDNN)", seed)
