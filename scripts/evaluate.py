"""Evaluate a trained PathoVision model on the test set.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --config configs/base.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.evaluation.report import generate_full_report
from src.models.factory import build_model
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PathoVision model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (if not in checkpoint)")
    parser.add_argument("--output-dir", type=str, default="results/02_baseline", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = get_device()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config")

    if config is None and args.config:
        from src.utils.config import load_config
        config = load_config(args.config)
    elif config is None:
        print("ERROR: No config found in checkpoint and --config not provided")
        sys.exit(1)

    # Build model and load weights
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Create test dataloader
    class_names = config["data"]["classes"]
    val_transform = get_val_transforms(config)

    dataset_dir = Path(config["data"]["dataset_dir"])
    test_dataset = SkinDiseaseDataset(
        root_dir=dataset_dir / "test",
        transform=val_transform,
        classes=class_names,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["data"].get("batch_size", 16),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    # Generate report
    report = generate_full_report(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )

    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
