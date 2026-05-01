"""Extract CNN features for hybrid CNN + classical ML pipeline."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_amp_dtype

logger = logging.getLogger(__name__)


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone features from a trained model.

    Uses model.extract_features() to get the GAP output before the
    classification head. For EfficientNetB2, this produces 1408-dim vectors.

    Args:
        model: Trained PathoVision model.
        dataloader: DataLoader (train, val, or test).
        device: Compute device.
        use_mixed_precision: Whether to use AMP.

    Returns:
        Tuple of (features, labels) as numpy arrays.
        features shape: (N, feature_dim)
        labels shape: (N,)
    """
    model.eval()
    amp_dtype = get_amp_dtype(device)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_mixed_precision):
                features = model.extract_features(images)

            all_features.append(features.float().cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    logger.info("Extracted features: shape=%s, labels=%s", features.shape, labels.shape)
    return features, labels


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save extracted features and labels to a .npz file.

    Args:
        features: Feature array (N, D).
        labels: Label array (N,).
        save_path: Path to save .npz file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, features=features, labels=labels)
    logger.info("Features saved to %s", save_path)


def load_features(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a .npz file.

    Args:
        path: Path to .npz file.

    Returns:
        Tuple of (features, labels).
    """
    data = np.load(path)
    return data["features"], data["labels"]
