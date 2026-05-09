"""Compare multiple trained models side by side.

Usage:
    python scripts/compare_models.py --checkpoints outputs/checkpoints/baseline.pt outputs/checkpoints/improved.pt
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import compute_full_report
from src.evaluation.report import run_inference
from src.models.factory import build_model
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths")
    parser.add_argument("--names", nargs="+", default=None, help="Model names for display")
    parser.add_argument("--output", type=str, default="results/03_multi_architecture/comparison.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = get_device()

    names = args.names or [Path(p).stem for p in args.checkpoints]
    all_results = []

    for ckpt_path, name in zip(args.checkpoints, names):
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({ckpt_path})")
        print(f"{'='*60}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        model = build_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        class_names = config["data"]["classes"]
        transform = get_val_transforms(config)
        dataset_dir = Path(config["data"]["dataset_dir"])

        test_dataset = SkinDiseaseDataset(
            root_dir=dataset_dir / "test", transform=transform, classes=class_names,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
        )

        y_true, y_pred, _ = run_inference(model, test_loader, device)
        report = compute_full_report(y_true, y_pred, class_names)

        row = {"Model": name, **report["global"]}
        # Add per-class F1
        for _, cls_row in report["per_class"].iterrows():
            row[f"F1_{cls_row['Class']}"] = cls_row["F1-Score"]

        all_results.append(row)

    # Comparison table
    df = pd.DataFrame(all_results)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(df.to_string(index=False, float_format="%.2f"))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
