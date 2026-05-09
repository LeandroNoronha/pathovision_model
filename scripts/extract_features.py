"""Extract CNN features for the hybrid pipeline.

Usage:
    python scripts/extract_features.py --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.hybrid.feature_extractor import extract_features, save_features
from src.models.factory import build_model
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CNN features")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/features")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    transform = get_val_transforms(config)
    class_names = config["data"]["classes"]
    dataset_dir = Path(config["data"]["dataset_dir"])
    output_dir = Path(args.output_dir)

    for split in ["train", "val", "test"]:
        print(f"\nExtracting features for {split}...")
        dataset = SkinDiseaseDataset(
            root_dir=dataset_dir / split,
            transform=transform,
            classes=class_names,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
        )

        features, labels = extract_features(model, loader, device)
        save_features(features, labels, output_dir / f"{split}_features.npz")
        print(f"  {split}: {features.shape[0]} samples, {features.shape[1]} features")

    print(f"\nFeatures saved to {output_dir}/")


if __name__ == "__main__":
    main()
