"""[R5] External validation — evaluate model on independent datasets.

Usage:
    python scripts/external_validation.py --checkpoint outputs/checkpoints/best.pt --external-dir datasets/external/fitzpatrick17k
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.data.dataset import SkinDiseaseDataset
from src.data.transforms import get_val_transforms
from src.evaluation.metrics import compute_full_report, compare_with_paper_baseline
from src.evaluation.report import run_inference
from src.models.factory import build_model
from src.utils.device import get_device


# Mapping from external dataset class names to PathoVision classes
EXTERNAL_CLASS_MAPPINGS = {
    "fitzpatrick17k": {
        "acne": "Acne",
        "candidiasis": "Candidiasis",
        "eczema": "Eczema",
        "atopic-dermatitis": "Eczema",
        "onychomycosis": "NailFungus",
        "nail-fungus": "NailFungus",
        "psoriasis": "Psoriasis",
        "tinea": "Tinea",
        "dermatophytosis": "Tinea",
        "ringworm": "Tinea",
    },
    "pad_ufes_20": {
        "ACK": "Acne",  # Actinic keratosis — not exact match
        "BCC": None,     # Skip — not in our classes
        "MEL": None,     # Skip
        "NEV": None,     # Skip
        "SCC": None,     # Skip
        "SEK": None,     # Skip
    },
}

PATHOVISION_CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]


def main() -> None:
    parser = argparse.ArgumentParser(description="[R5] External validation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--external-dir", type=str, required=True, help="External dataset dir (class/img)")
    parser.add_argument("--dataset-name", type=str, default="external", help="Dataset name for reporting")
    parser.add_argument("--output-dir", type=str, default="results/08_external_validation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Create external dataset
    transform = get_val_transforms(config)
    external_dir = Path(args.external_dir)

    # Discover available classes
    available_classes = []
    mapping = EXTERNAL_CLASS_MAPPINGS.get(args.dataset_name, {})

    for subdir in sorted(external_dir.iterdir()):
        if not subdir.is_dir():
            continue
        mapped = mapping.get(subdir.name.lower(), subdir.name)
        if mapped and mapped in PATHOVISION_CLASSES:
            available_classes.append(subdir.name)
            print(f"  Found: {subdir.name} -> {mapped}")

    if not available_classes:
        # Try direct class names
        for subdir in sorted(external_dir.iterdir()):
            if subdir.is_dir() and subdir.name in PATHOVISION_CLASSES:
                available_classes.append(subdir.name)

    if not available_classes:
        print("ERROR: No compatible classes found in external dataset.")
        print(f"Expected class directories in: {external_dir}")
        return

    # Load external dataset with mapping
    external_dataset = SkinDiseaseDataset(
        root_dir=external_dir,
        transform=transform,
        classes=available_classes,
    )

    external_loader = torch.utils.data.DataLoader(
        external_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Run inference
    print(f"\n=== External Validation: {args.dataset_name} ===")
    print(f"Images: {len(external_dataset)}, Classes: {available_classes}")

    y_true, y_pred, y_proba = run_inference(model, external_loader, device)

    # Compute metrics
    report = compute_full_report(y_true, y_pred, available_classes)

    print(f"\nExternal Validation Results ({args.dataset_name}):")
    print(f"  Global Accuracy: {report['global']['global_accuracy']:.2f}%")
    print(f"  Weighted F1:     {report['global']['f1_weighted']:.2f}%")
    print(f"\n{report['classification_report']}")

    # Save
    report["per_class"].to_csv(output_dir / f"{args.dataset_name}_per_class.csv", index=False)

    # Comparison table: Internal vs External
    print("\n=== Internal vs External Comparison ===")
    print("(Run scripts/evaluate.py first to get internal metrics)")
    print(f"\nExternal results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
