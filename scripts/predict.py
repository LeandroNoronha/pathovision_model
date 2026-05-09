"""Predict skin disease from a single image.

Usage:
    python scripts/predict.py --checkpoint outputs/checkpoints/best.pt --image path/to/image.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch

from src.data.transforms import get_val_transforms
from src.models.factory import build_model
from src.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict skin disease from image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--top-k", type=int, default=3, help="Show top-K predictions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    device = get_device()

    # Load checkpoint and model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    class_names = config["data"]["classes"]

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    image = cv2.imread(args.image)
    if image is None:
        print(f"ERROR: Could not load image: {args.image}")
        sys.exit(1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_val_transforms(config)
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    result = model.predict(input_tensor)
    probas = result["probabilities"][0].cpu().numpy()

    # Sort by probability
    sorted_indices = probas.argsort()[::-1]

    print(f"\nImage: {args.image}")
    print(f"{'='*40}")
    print(f"{'Class':<20} {'Confidence':>10}")
    print(f"{'-'*40}")
    for i in range(min(args.top_k, len(class_names))):
        idx = sorted_indices[i]
        print(f"{class_names[idx]:<20} {probas[idx]*100:>9.2f}%")

    predicted = class_names[sorted_indices[0]]
    confidence = probas[sorted_indices[0]] * 100
    print(f"\nPrediction: {predicted} ({confidence:.1f}%)")


if __name__ == "__main__":
    main()
