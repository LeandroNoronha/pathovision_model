"""Balanced sampling utilities for handling class imbalance."""

import logging
from collections import Counter

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


def get_balanced_sampler(dataset: "SkinDiseaseDataset") -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that oversamples minority classes.

    Each sample gets a weight inversely proportional to its class frequency,
    so all classes are effectively sampled equally per epoch.

    Args:
        dataset: SkinDiseaseDataset instance.

    Returns:
        WeightedRandomSampler for the DataLoader.
    """
    labels = dataset.get_labels()
    class_counts = Counter(labels)
    num_samples = len(labels)

    # Inverse frequency weights per class
    class_weights = {
        cls: num_samples / count for cls, count in class_counts.items()
    }

    # Assign weight to each sample
    sample_weights = torch.tensor(
        [class_weights[label] for label in labels],
        dtype=torch.float64,
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )

    # Log effective distribution
    for cls_idx, count in sorted(class_counts.items()):
        weight = class_weights[cls_idx]
        cls_name = dataset.idx_to_class[cls_idx]
        logger.info(
            "Class '%s': %d samples, weight=%.2f",
            cls_name, count, weight,
        )

    return sampler


def compute_class_weights(
    dataset: "SkinDiseaseDataset",
    method: str = "inverse_frequency",
) -> torch.Tensor:
    """Compute class weights for the loss function.

    Args:
        dataset: SkinDiseaseDataset instance.
        method: Weighting method. Options:
            - "inverse_frequency": weight = N / (n_classes * n_class)
            - "effective_samples": uses effective number of samples (beta=0.9999)
            - "sqrt_inverse": weight = sqrt(N / n_class)

    Returns:
        Tensor of class weights, shape (num_classes,).
    """
    labels = dataset.get_labels()
    class_counts = Counter(labels)
    num_classes = len(dataset.classes)
    total = len(labels)

    # Build counts array in class order
    counts = np.array([class_counts.get(i, 1) for i in range(num_classes)], dtype=np.float64)

    if method == "inverse_frequency":
        weights = total / (num_classes * counts)
    elif method == "effective_samples":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes  # normalize
    elif method == "sqrt_inverse":
        weights = np.sqrt(total / counts)
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown class weight method: {method}")

    logger.info("Class weights (%s): %s", method, dict(zip(dataset.classes, weights)))

    return torch.tensor(weights, dtype=torch.float32)
