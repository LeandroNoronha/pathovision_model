"""PyTorch Dataset for dermatological image classification."""

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SkinDiseaseDataset(Dataset):
    """Dataset for loading skin disease images from a directory structure.

    Expected directory layout:
        dataset_dir/
            split/              # train, val, or test
                ClassName/
                    image1.jpg
                    image2.png
                    ...

    Args:
        root_dir: Root directory of the dataset split.
        transform: Albumentations transform pipeline.
        classes: Ordered list of class names. If None, inferred from subdirectories.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: A.Compose | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # Discover or use provided class list
        if classes is not None:
            self.classes = classes
        else:
            self.classes = sorted([
                d.name for d in self.root_dir.iterdir() if d.is_dir()
            ])

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # Collect all image paths and labels
        self.samples: list[tuple[Path, int]] = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                logger.warning("Class directory not found: %s", cls_dir)
                continue

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        logger.info(
            "Loaded %d images across %d classes from %s",
            len(self.samples), len(self.classes), self.root_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dictionary with keys:
                - image: Tensor (C, H, W)
                - label: Integer class index
                - class_name: String class name
                - path: String image file path
        """
        img_path, label = self.samples[idx]

        # Load image as RGB using OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "label": label,
            "class_name": self.idx_to_class[label],
            "path": str(img_path),
        }

    def get_class_distribution(self) -> dict[str, int]:
        """Get the number of samples per class.

        Returns:
            Dictionary mapping class name to count.
        """
        dist: dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            dist[self.idx_to_class[label]] += 1
        return dist

    def get_labels(self) -> list[int]:
        """Get all labels as a list (for sampler construction).

        Returns:
            List of integer labels.
        """
        return [label for _, label in self.samples]


def create_dataloaders(
    config: dict[str, Any],
    train_transform: A.Compose,
    val_transform: A.Compose,
) -> dict[str, torch.utils.data.DataLoader]:
    """Create train, validation, and test DataLoaders from config.

    Args:
        config: Configuration dictionary.
        train_transform: Transform for training data.
        val_transform: Transform for validation and test data.

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoader instances.
    """
    from src.data.sampler import get_balanced_sampler

    data_cfg = config["data"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    classes = data_cfg.get("classes")
    batch_size = data_cfg.get("batch_size", 16)
    num_workers = data_cfg.get("num_workers", 4)
    pin_memory = data_cfg.get("pin_memory", True)

    # Create datasets
    train_dataset = SkinDiseaseDataset(
        root_dir=dataset_dir / "train",
        transform=train_transform,
        classes=classes,
    )
    val_dataset = SkinDiseaseDataset(
        root_dir=dataset_dir / "val",
        transform=val_transform,
        classes=classes,
    )
    test_dataset = SkinDiseaseDataset(
        root_dir=dataset_dir / "test",
        transform=val_transform,
        classes=classes,
    )

    # Log class distribution
    train_dist = train_dataset.get_class_distribution()
    logger.info("Training class distribution: %s", train_dist)

    # Build sampler for balanced training (if configured)
    train_sampler = None
    train_shuffle = True
    if data_cfg.get("balanced_sampling", False):
        train_sampler = get_balanced_sampler(train_dataset)
        train_shuffle = False  # sampler handles shuffling
        logger.info("Using balanced sampling for training")

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    for split, dl in dataloaders.items():
        logger.info(
            "%s: %d samples, %d batches",
            split, len(dl.dataset), len(dl),
        )

    return dataloaders
