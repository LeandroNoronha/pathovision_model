# Dataset Download and Preparation

## Sources

This project uses two publicly available datasets from Kaggle:

1. **Human Skin Diseases -- A Complete Dataset**
   - Author: Youssef Mohamed
   - URL: https://www.kaggle.com/datasets/youssefmohmmed/human-skin-diseases-a-complete-dataset
   - Contains clinical images of multiple skin conditions

2. **Dermnet**
   - Author: Shubham Goel (originally curated by Bill Hall)
   - URL: https://www.kaggle.com/datasets/shubhamgoel27/dermnet
   - Dermatology image archive with diverse conditions

## Prerequisites

- A Kaggle account and API key placed at `~/.kaggle/kaggle.json`
- Install the Kaggle CLI: `pip install kaggle`

## Step-by-Step Preparation

### 1. Download raw datasets

```bash
python datasets/download_kaggle.py
```

Downloads both datasets into `datasets/raw/`.

### 2. Organize into 7-class structure

```bash
python datasets/organize_dataset.py
```

Merges both sources, maps condition labels to the 7 target classes (Acne, Candidiasis, Eczema, NailFungus, Normal, Psoriasis, Tinea), and creates the train/val/test split (76.1% / 8.4% / 15.6%).

Output: `datasets/merged/`

### 3. Detect and remove duplicates

```bash
python datasets/detect_duplicates.py --dataset-dir datasets/merged
```

Uses perceptual hashing (pHash) to identify duplicate and near-duplicate images, including cross-split leakage (same image appearing in both train and test). Removes 4,071 duplicates including 1,017 cross-split leakage groups.

### 4. Verify dataset integrity

```bash
python datasets/verify_dataset.py --dataset-dir datasets/data
```

Checks:
- All images are readable (valid JPEG/PNG)
- Class distribution matches expectations
- No remaining duplicates across splits
- Split ratios are correct

## Final Dataset Summary

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 7,028  | 76.1%      |
| Val   | 775    | 8.4%       |
| Test  | 1,424  | 15.6%      |
| **Total** | **9,227** | **100%** |

## Important Notes

- Image files are **not** committed to git (gitignored)
- All dataset paths are config-driven via YAML files in `configs/`
- Augmentation is applied **only** to the training set, never to val/test
- The dataset quality control report is saved to `results/01_dataset_qc/`
