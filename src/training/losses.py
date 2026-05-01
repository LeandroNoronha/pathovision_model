"""Loss functions for skin disease classification."""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Downweights well-classified examples, focusing training on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (0 = standard CE, 2 = recommended).
        alpha: Class weights tensor or None for uniform.
        label_smoothing: Label smoothing factor.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output logits (batch, num_classes).
            targets: Ground truth class indices (batch,).

        Returns:
            Scalar loss value.
        """
        num_classes = logits.shape[1]

        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Compute probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight
        focal_weight = (1.0 - probs) ** self.gamma

        # Class weights
        if self.alpha is not None:
            alpha_weight = self.alpha[targets].unsqueeze(1)
            focal_weight = focal_weight * alpha_weight

        # Focal loss
        loss = -focal_weight * smooth_targets * log_probs
        return loss.sum(dim=1).mean()


def build_loss(config: dict[str, Any], class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build loss function from configuration.

    Args:
        config: Configuration dictionary with 'training.loss' key.
        class_weights: Optional class weight tensor.

    Returns:
        Loss function module.
    """
    loss_cfg = config["training"]["loss"]
    loss_name = loss_cfg.get("name", "cross_entropy")
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)

    if loss_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Loss: CrossEntropy (label_smoothing=%.2f, class_weights=%s)",
            label_smoothing, class_weights is not None,
        )

    elif loss_name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        loss_fn = FocalLoss(
            gamma=gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )
        logger.info(
            "Loss: FocalLoss (gamma=%.1f, label_smoothing=%.2f, class_weights=%s)",
            gamma, label_smoothing, class_weights is not None,
        )

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    return loss_fn
