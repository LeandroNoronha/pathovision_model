"""Verify dataset integrity and print distribution statistics.

Usage:
    python datasets/verify_dataset.py
    python datasets/verify_dataset.py --dataset-dir datasets/merged
"""

import argparse
from pathlib import Path

import cv2

TARGET_CLASSES = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Expected counts from the paper (Table F.5)
PAPER_COUNTS = {
    "Acne":        {"train": 1301, "val": 132, "test": 377, "total": 1810},
    "Candidiasis": {"train": 248,  "val": 96,  "test": 27,  "total": 371},
    "Eczema":      {"train": 2017, "val": 228, "test": 421, "total": 2666},
    "NailFungus":  {"train": 832,  "val": 208, "test": 261, "total": 1301},
    "Normal":      {"train": 1526, "val": 120, "test": 189, "total": 1835},
    "Psoriasis":   {"train": 2141, "val": 84,  "test": 440, "total": 2665},
    "Tinea":       {"train": 815,  "val": 108, "test": 102, "total": 1025},
}
PAPER_TOTAL = 11673


def count_images(dataset_dir: Path) -> dict[str, dict[str, int]]:
    """Count images per class per split."""
    counts: dict[str, dict[str, int]] = {}

    for cls in TARGET_CLASSES:
        counts[cls] = {}
        for split in ["train", "val", "test"]:
            cls_dir = dataset_dir / split / cls
            if cls_dir.exists():
                n = len([f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])
            else:
                n = 0
            counts[cls][split] = n
        counts[cls]["total"] = sum(counts[cls].values())

    return counts


def check_corrupt_images(dataset_dir: Path, max_check: int = 0) -> list[str]:
    """Try to open each image and detect corrupt files.

    Args:
        dataset_dir: Root of the merged dataset.
        max_check: Max images to check (0 = all).

    Returns:
        List of corrupt image paths.
    """
    corrupt = []
    checked = 0

    for split in ["train", "val", "test"]:
        for cls in TARGET_CLASSES:
            cls_dir = dataset_dir / split / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                if max_check > 0 and checked >= max_check:
                    return corrupt

                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt.append(str(img_path))
                checked += 1

    return corrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PathoVision dataset")
    parser.add_argument("--dataset-dir", type=str, default="datasets/merged")
    parser.add_argument("--check-corrupt", action="store_true", help="Check for corrupt images")
    parser.add_argument("--max-check", type=int, default=0, help="Max images to check for corruption (0=all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        print("Run 'python datasets/organize_dataset.py' first.")
        return

    # Count images
    counts = count_images(dataset_dir)

    # Print distribution
    grand_total = 0
    print(f"\n{'Class':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}  {'Paper':>7}  {'Match':>5}")
    print("=" * 72)

    for cls in TARGET_CLASSES:
        c = counts[cls]
        paper = PAPER_COUNTS[cls]
        match = "OK" if c["total"] == paper["total"] else "DIFF"
        grand_total += c["total"]
        print(
            f"{cls:<15} {c['train']:>7} {c['val']:>7} {c['test']:>7} {c['total']:>7}  "
            f"{paper['total']:>7}  {match:>5}"
        )

    print("=" * 72)
    paper_match = "OK" if grand_total == PAPER_TOTAL else "DIFF"
    print(f"{'TOTAL':<15} {'':>7} {'':>7} {'':>7} {grand_total:>7}  {PAPER_TOTAL:>7}  {paper_match:>5}")

    # Split totals
    for split in ["train", "val", "test"]:
        split_total = sum(counts[cls][split] for cls in TARGET_CLASSES)
        pct = (split_total / grand_total * 100) if grand_total > 0 else 0
        print(f"  {split}: {split_total} ({pct:.1f}%)")

    # Check for corrupt images
    if args.check_corrupt:
        print("\nChecking for corrupt images...")
        corrupt = check_corrupt_images(dataset_dir, args.max_check)
        if corrupt:
            print(f"  WARNING: {len(corrupt)} corrupt images found:")
            for p in corrupt[:20]:
                print(f"    {p}")
        else:
            print("  All images OK.")


if __name__ == "__main__":
    main()
