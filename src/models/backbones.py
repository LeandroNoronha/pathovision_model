"""Backbone registry for image classification models.

Uses the `timm` library for pretrained model loading.
All backbones return feature tensors (before classification head).
"""

import logging
from typing import Any

import timm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Registry: backbone name -> timm model name
BACKBONE_REGISTRY: dict[str, str] = {
    # EfficientNet family
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b4": "efficientnet_b4",
    # EfficientNetV2 family
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "efficientnetv2_m": "tf_efficientnetv2_m",
    # ConvNeXt family
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    # Vision Transformers (patch16 supports flexible input via img_size param)
    "vit_small": "vit_small_patch16_384",
    "vit_base": "vit_base_patch16_384",
    # Swin Transformers (use 384 variant for flexible input)
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    # DeiT (Data-efficient Image Transformer)
    "deit_small": "deit_small_patch16_224",
}


def get_backbone(
    name: str,
    pretrained: bool = True,
    num_classes: int = 0,
) -> nn.Module:
    """Create a backbone model from the registry.

    Args:
        name: Backbone name (key in BACKBONE_REGISTRY).
        pretrained: Whether to load pretrained (ImageNet) weights.
        num_classes: Number of output classes (0 = feature extractor only).

    Returns:
        timm model instance.

    Raises:
        ValueError: If backbone name is not in registry.
    """
    if name not in BACKBONE_REGISTRY:
        available = ", ".join(sorted(BACKBONE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {available}"
        )

    timm_name = BACKBONE_REGISTRY[name]
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Created backbone '%s' (timm: %s), pretrained=%s, params=%.2fM",
        name, timm_name, pretrained, num_params / 1e6,
    )

    return model


def get_backbone_feature_dim(name: str) -> int:
    """Get the feature dimension of a backbone's output.

    Args:
        name: Backbone name.

    Returns:
        Number of output features from the backbone.
    """
    timm_name = BACKBONE_REGISTRY[name]
    model = timm.create_model(timm_name, pretrained=False, num_classes=0)

    # Get feature dim from a dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        features = model(dummy)
        feature_dim = features.shape[-1]

    del model
    return feature_dim


def freeze_backbone(
    model: nn.Module,
    strategy: str = "last_n",
    freeze_except_last_n: int = 10,
) -> tuple[int, int]:
    """Freeze backbone layers according to strategy.

    Args:
        model: The backbone model.
        strategy: Freezing strategy. Options:
            - "last_n": Freeze all except last N parameters/layers
            - "all": Freeze all layers
            - "none": Freeze nothing
        freeze_except_last_n: Number of layer groups to keep unfrozen.

    Returns:
        Tuple of (frozen_params, trainable_params).
    """
    all_params = list(model.parameters())

    if strategy == "all":
        for p in all_params:
            p.requires_grad = False
    elif strategy == "none":
        for p in all_params:
            p.requires_grad = True
    elif strategy == "last_n":
        # Freeze all parameters first
        for p in all_params:
            p.requires_grad = False
        # Unfreeze the last N parameter groups
        params_to_unfreeze = all_params[-freeze_except_last_n:]
        for p in params_to_unfreeze:
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    frozen = sum(1 for p in all_params if not p.requires_grad)
    trainable = sum(1 for p in all_params if p.requires_grad)
    frozen_params = sum(p.numel() for p in all_params if not p.requires_grad)
    trainable_params = sum(p.numel() for p in all_params if p.requires_grad)

    logger.info(
        "Backbone freezing: %d/%d layers frozen (%.2fM frozen, %.2fM trainable)",
        frozen, frozen + trainable, frozen_params / 1e6, trainable_params / 1e6,
    )

    return frozen_params, trainable_params


def list_available_backbones() -> list[str]:
    """List all available backbone names."""
    return sorted(BACKBONE_REGISTRY.keys())
