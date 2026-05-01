"""[R5] Download external validation datasets.

External datasets for independent validation:
- Fitzpatrick17k: 16,577 images, 114 conditions (filter to 7 target classes)
- PAD-UFES-20: 2,298 images, 6 classes (partial overlap)

Usage:
    python datasets/download_external.py --dataset fitzpatrick17k
    python datasets/download_external.py --manual
"""

import argparse
import sys
from pathlib import Path

EXTERNAL_DATASETS = {
    "fitzpatrick17k": {
        "source": "GitHub + Kaggle",
        "url": "https://github.com/mattgroh/fitzpatrick17k",
        "kaggle_id": None,
        "description": "16,577 dermatology images with Fitzpatrick skin type labels",
        "relevant_classes": ["acne", "eczema", "psoriasis", "tinea", "candidiasis", "onychomycosis"],
    },
    "pad_ufes_20": {
        "source": "Mendeley Data",
        "url": "https://data.mendeley.com/datasets/zr7vgbcyr2/1",
        "kaggle_id": None,
        "description": "2,298 images from Brazilian dermatology clinic (smartphone captured)",
        "relevant_classes": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
    },
    "dermnet_nz": {
        "source": "DermNet NZ (web scraping required)",
        "url": "https://dermnetnz.org/image-library",
        "kaggle_id": None,
        "description": "Large dermatology image library, subset different from training data",
        "relevant_classes": ["acne", "eczema", "psoriasis", "tinea", "candidiasis"],
    },
}

RAW_DIR = Path(__file__).parent / "external"


def print_instructions() -> None:
    """Print download instructions for all external datasets."""
    print("\n" + "=" * 70)
    print("[R5] EXTERNAL VALIDATION DATASETS — Download Instructions")
    print("=" * 70)

    for name, info in EXTERNAL_DATASETS.items():
        print(f"\n--- {name} ---")
        print(f"  Source: {info['source']}")
        print(f"  URL: {info['url']}")
        print(f"  Description: {info['description']}")
        print(f"  Relevant classes: {', '.join(info['relevant_classes'])}")
        print(f"  Extract to: {RAW_DIR / name}/")

    print(f"\n{'='*70}")
    print("RECOMMENDED: Start with Fitzpatrick17k (has skin type labels for fairness)")
    print(f"{'='*70}")

    print(f"\nAfter downloading, organize class directories under:")
    print(f"  {RAW_DIR}/<dataset_name>/<class_name>/images...")
    print(f"\nThen run: python scripts/external_validation.py --checkpoint <path> --external-dir {RAW_DIR}/<dataset>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external validation datasets")
    parser.add_argument("--dataset", type=str, choices=list(EXTERNAL_DATASETS.keys()), default=None)
    parser.add_argument("--manual", action="store_true", help="Show manual instructions")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.manual or args.dataset is None:
        print_instructions()
    else:
        info = EXTERNAL_DATASETS[args.dataset]
        print(f"Dataset: {args.dataset}")
        print(f"Please download manually from: {info['url']}")
        print(f"Extract to: {RAW_DIR / args.dataset}/")


if __name__ == "__main__":
    main()
