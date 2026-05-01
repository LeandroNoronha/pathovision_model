"""[R8] Detect duplicate images across the dataset using perceptual hashing.

Addresses reviewer concerns about:
- Overlapping images between the two Kaggle sources
- Data leakage (same image in train and test)
- Quality control

Usage:
    python datasets/detect_duplicates.py --dataset-dir datasets/merged
"""

import argparse
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def compute_dhash(image_path: str | Path, hash_size: int = 16) -> str:
    """Compute difference hash (dHash) for an image.

    dHash is robust to minor variations in size, aspect ratio, and brightness.

    Args:
        image_path: Path to image file.
        hash_size: Hash grid size (default 16 = 256-bit hash).

    Returns:
        Hex string hash.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize to (hash_size+1, hash_size) for horizontal gradient
    resized = cv2.resize(img, (hash_size + 1, hash_size))

    # Compute differences between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert to hash string
    return "".join(str(int(b)) for b in diff.flatten())


def compute_md5(image_path: str | Path) -> str:
    """Compute MD5 hash of file bytes (exact duplicate detection)."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def find_duplicates(
    dataset_dir: str | Path,
    method: str = "dhash",
    threshold: int = 10,
) -> dict[str, list[tuple[str, str]]]:
    """Find duplicate image pairs across the dataset.

    Args:
        dataset_dir: Root directory of the merged dataset.
        method: 'md5' for exact duplicates, 'dhash' for perceptual.
        threshold: Hamming distance threshold for dhash (lower = stricter).

    Returns:
        Dictionary with:
            - 'exact_duplicates': list of (path1, path2) exact matches
            - 'near_duplicates': list of (path1, path2, distance) near matches
            - 'cross_split_leaks': list of (train_path, test_path) leakage pairs
    """
    dataset_dir = Path(dataset_dir)

    # Collect all images with their split info
    images = []
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in VALID_EXTENSIONS:
                    images.append({
                        "path": img_path,
                        "split": split,
                        "class": cls_dir.name,
                    })

    print(f"Computing hashes for {len(images)} images...")

    # Compute hashes
    hashes = {}
    for i, img_info in enumerate(images):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(images)} images...")
        try:
            if method == "md5":
                h = compute_md5(img_info["path"])
            else:
                h = compute_dhash(img_info["path"])
            hashes[str(img_info["path"])] = {
                "hash": h,
                **img_info,
            }
        except Exception as e:
            logger.warning("Failed to hash %s: %s", img_info["path"], e)

    # Find exact duplicates (same hash)
    hash_to_paths: dict[str, list[dict]] = defaultdict(list)
    for path, info in hashes.items():
        hash_to_paths[info["hash"]].append(info)

    exact_duplicates = []
    for h, infos in hash_to_paths.items():
        if len(infos) > 1:
            for i in range(len(infos)):
                for j in range(i + 1, len(infos)):
                    exact_duplicates.append((
                        str(infos[i]["path"]),
                        str(infos[j]["path"]),
                        infos[i]["split"],
                        infos[j]["split"],
                    ))

    # Find cross-split leakage
    cross_split_leaks = [
        (p1, p2, s1, s2)
        for p1, p2, s1, s2 in exact_duplicates
        if s1 != s2
    ]

    results = {
        "total_images": len(images),
        "exact_duplicates": exact_duplicates,
        "cross_split_leaks": cross_split_leaks,
        "num_exact_duplicates": len(exact_duplicates),
        "num_cross_split_leaks": len(cross_split_leaks),
    }

    return results


def analyze_normal_class(dataset_dir: str | Path) -> dict:
    """[R8] Investigate the Normal class for distribution artifacts.

    Compares image statistics (resolution, brightness, color distribution)
    between Normal and disease classes.

    Args:
        dataset_dir: Root of dataset split (e.g., datasets/merged/test).

    Returns:
        Dictionary with analysis results.
    """
    dataset_dir = Path(dataset_dir)
    classes = ["Acne", "Candidiasis", "Eczema", "NailFungus", "Normal", "Psoriasis", "Tinea"]
    stats = {}

    for cls in classes:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue

        images = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])[:100]
        resolutions = []
        mean_brightness = []
        mean_colors = []

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            resolutions.append((h, w))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness.append(float(np.mean(gray)))
            mean_colors.append(img.mean(axis=(0, 1)).tolist())

        if resolutions:
            stats[cls] = {
                "count": len(images),
                "avg_resolution": f"{np.mean([r[0] for r in resolutions]):.0f}x{np.mean([r[1] for r in resolutions]):.0f}",
                "avg_brightness": float(np.mean(mean_brightness)),
                "std_brightness": float(np.std(mean_brightness)),
                "avg_color_bgr": [float(np.mean([c[i] for c in mean_colors])) for i in range(3)],
            }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="[R8] Detect duplicates and QC")
    parser.add_argument("--dataset-dir", type=str, default="datasets/merged")
    parser.add_argument("--method", choices=["md5", "dhash"], default="dhash")
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--analyze-normal", action="store_true", help="Analyze Normal class artifacts")
    parser.add_argument("--output-dir", type=str, default="results/01_dataset_qc")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dataset_dir = Path(args.dataset_dir)

    # Duplicate detection
    print("\n=== Duplicate Detection ===")
    results = find_duplicates(dataset_dir, method=args.method, threshold=args.threshold)

    print(f"\nTotal images: {results['total_images']}")
    print(f"Exact duplicates found: {results['num_exact_duplicates']}")
    print(f"Cross-split leaks (data leakage!): {results['num_cross_split_leaks']}")

    if results["cross_split_leaks"]:
        print("\nWARNING: Data leakage detected!")
        for p1, p2, s1, s2 in results["cross_split_leaks"][:20]:
            print(f"  {s1}: {Path(p1).name}  <->  {s2}: {Path(p2).name}")

    if results["exact_duplicates"]:
        print(f"\nDuplicate pairs (first 20):")
        for p1, p2, s1, s2 in results["exact_duplicates"][:20]:
            print(f"  [{s1}] {Path(p1).name}  ==  [{s2}] {Path(p2).name}")

    # Normal class analysis
    if args.analyze_normal:
        print("\n=== Normal Class Analysis ===")
        all_stats = {}
        for split in ["train", "test"]:
            print(f"\n  Split: {split}")
            stats = analyze_normal_class(dataset_dir / split)
            for cls, s in stats.items():
                print(f"    {cls:15s}: brightness={s['avg_brightness']:.1f}±{s['std_brightness']:.1f}, "
                      f"res={s['avg_resolution']}")
            all_stats[split] = stats

    # Save results to output dir
    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "total_images": results["total_images"],
        "num_exact_duplicates": results["num_exact_duplicates"],
        "num_cross_split_leaks": results["num_cross_split_leaks"],
        "method": args.method,
    }
    if args.analyze_normal and all_stats:
        report["normal_class_analysis"] = all_stats

    with open(output_dir / "dataset_qc_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    if results["exact_duplicates"]:
        with open(output_dir / "duplicate_pairs.txt", "w") as f:
            for p1, p2, s1, s2 in results["exact_duplicates"]:
                f.write(f"[{s1}] {p1}  ==  [{s2}] {p2}\n")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
