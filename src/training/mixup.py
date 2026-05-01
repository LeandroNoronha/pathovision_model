"""Mixup and CutMix data augmentation for training regularization."""

import numpy as np
import torch


def mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup augmentation to a batch.

    Blends pairs of images and their labels with a random mixing coefficient.

    Args:
        images: Batch of images (B, C, H, W).
        labels: Batch of labels (B,).
        alpha: Beta distribution parameter (higher = more mixing).

    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lam).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for Mixup/CutMix.

    Args:
        criterion: Base loss function.
        logits: Model output logits.
        labels_a: First set of labels.
        labels_b: Second set of labels.
        lam: Mixing coefficient.

    Returns:
        Weighted loss.
    """
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
