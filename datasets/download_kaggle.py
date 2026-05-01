"""Download and extract the two Kaggle datasets used by PathoVision.

Datasets:
    1. "Human Skin Diseases" by Youssef Mohamed
       https://www.kaggle.com/datasets/youssefmohmmed/human-skin-diseases-image
    2. "Dermnet" by Shubham Goel / Bill Hall
       https://www.kaggle.com/datasets/shubhamgoel27/dermnet

Requirements:
    - kaggle CLI installed and configured (~/.kaggle/kaggle.json)
    - Or manual download: place zip files in datasets/raw/

Usage:
    python datasets/download_kaggle.py
    python datasets/download_kaggle.py --manual  # skip download, just extract
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASETS = {
    "human_skin_diseases": {
        "kaggle_id": "youssefmohmmed/human-skin-diseases-image",
        "filename": "human-skin-diseases-image.zip",
    },
    "dermnet": {
        "kaggle_id": "shubhamgoel27/dermnet",
        "filename": "dermnet.zip",
    },
}

RAW_DIR = Path(__file__).parent / "raw"


def download_datasets() -> None:
    """Download datasets using kaggle CLI."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Downloading: {name}")
        print(f"Kaggle ID:   {info['kaggle_id']}")
        print(f"{'='*60}")

        dest = RAW_DIR / name
        if dest.exists():
            print(f"  Already exists at {dest}, skipping download.")
            continue

        try:
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "-d", info["kaggle_id"],
                    "-p", str(RAW_DIR),
                    "--unzip",
                ],
                check=True,
            )
            print(f"  Downloaded and extracted to {RAW_DIR}")
        except FileNotFoundError:
            print("  ERROR: 'kaggle' CLI not found.")
            print("  Install with: pip install kaggle")
            print("  Configure: place kaggle.json in ~/.kaggle/")
            print(f"  Or download manually and place in {RAW_DIR}")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Download failed: {e}")
            print(f"  Download manually from: https://www.kaggle.com/datasets/{info['kaggle_id']}")
            print(f"  Place files in: {RAW_DIR}")
            sys.exit(1)


def print_manual_instructions() -> None:
    """Print instructions for manual download."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print()
    for name, info in DATASETS.items():
        url = f"https://www.kaggle.com/datasets/{info['kaggle_id']}"
        print(f"  {name}:")
        print(f"    URL: {url}")
        print(f"    Extract to: {RAW_DIR / name}")
        print()
    print(f"After downloading, run: python datasets/organize_dataset.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PathoVision datasets from Kaggle")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions")
    args = parser.parse_args()

    if args.manual:
        print_manual_instructions()
    else:
        download_datasets()
        print("\nDone! Next step: python datasets/organize_dataset.py")


if __name__ == "__main__":
    main()
