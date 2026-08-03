# PathoVision: Multi-Architecture Deep Learning for Dermatological Image Classification

This repository contains the complete implementation and research materials for the PathoVision article.  
The project develops a comprehensive framework for automated classification of dermatological conditions using multiple state-of-the-art deep learning architectures, with emphasis on explainability, fairness evaluation, and rigorous experimental validation.

The repository serves as a comprehensive research workspace, encompassing data preprocessing pipelines, model implementations, experimental protocols, and publication-ready documentation, ensuring reproducibility and scientific rigor throughout the dermatological image analysis project.

## Versioning

The evolution of this project has generated different versions of the classifier:

- **v1.0.0.0 (Stable)**  
  - PathoVision v2: Complete rewrite targeting Python 3.13
  - Multi-architecture support (ViT, Swin Transformer, ConvNeXt)
  - Dataset cleaning pipeline (9,227 images after duplicate removal)
  - Explainability with Grad-CAM analysis
  - Fairness evaluation with ITA skin tone stratification
  - Recommended version for use.

- **v0.4.0.0 (Stable_old)**  
  - Better legibility.  
  - Improved acquisition of metrics.  

- **v0.3.0.0 (Stable_old)**  
  - First stable and functional version of the model.  
  - Trained with EfficientNetB2 and validated metrics.

- **v0.2.0.0 (Unstable)**  
  - Contains compatibility issues with Keras/TensorFlow.  
  - Not recommended for production use.

- **v0.1.0.0 (Broken/Deprecated)**  
  - Initial prototype.  
  - Does not work with current dependency versions.

### How to use a specific version

To check out the stable release, run:

```bash
git checkout tags/v1.0.0.0
```

## PathoVision v2 Overview

PathoVision v2 is a complete rewrite of the dermatological image classification framework targeting Python 3.13. The project compares ViT (DINOv2), Swin Transformer, and ConvNeXt architectures with balanced and non-balanced training strategies.

**Key Features:**
- **Dataset Pipeline**: Automated download, organization, duplicate detection, and validation
- **Multi-Architecture**: Support for ViT, Swin Transformer, ConvNeXt, and hybrid ensembles
- **Explainability**: Grad-CAM analysis for model interpretability
- **Fairness Evaluation**: ITA skin tone stratification for demographic analysis
- **Comprehensive Testing**: Full test suite with pytest configuration
- **Modern Tooling**: Ruff linting/formatting, type hints, and reproducible configurations

**Dataset Processing:**
- Sources: Human Skin Diseases (Kaggle) + Dermnet (Kaggle)
- Classes: 7 dermatological conditions (Acne, Candidiasis, Eczema, NailFungus, Normal, Psoriasis, Tinea)
- Final Dataset: 9,227 images (7.378 train / 920 val / 929 test)
- Duplicate Removal: dHash (difference hash, perceptual) deduplication removed exact and near-duplicate images and 1,017 train–test leakage groups

**Architectures Supported:**
- ViT (DINOv2) - Vision Transformer
- Swin Transformer - Hierarchical vision transformer
- ConvNeXt - Modern convolutional network
- Hybrid Ensemble - Feature extraction + classical ML

## Repository Structure

- `src/` - Core library code (data, models, training, evaluation)
- `scripts/` - Reproducible experiments (train.py, evaluate.py, train_all.py, generate_evidence.py)
- `configs/` - YAML training configurations (17 config files for different architectures)
- `datasets/` - Dataset organization, dedup, split and cleaning tools
- `tests/` - Test suite with pytest (9 tests)
- `evidence/` - Experimental results: prediction CSVs, figures, McNemar test, PDF report

## Quick Start

### Prerequisites
- Python 3.13+
- PyTorch 2.6+
- CUDA 12.4+ (recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Dataset Preparation
```bash
# Download datasets from Kaggle (requires kaggle.json token)
python datasets/download_kaggle.py

# Organize and clean dataset
python datasets/organize_dataset.py

# Detect and remove duplicates (dHash perceptual)
python datasets/detect_duplicates2.py --dataset-dir datasets/merged --method dhash --delete

# Create stratified 80/10/10 train/val/test split
python datasets/create_split.py --input datasets/merged --output datasets/data --seed 42

# Verify final dataset
python datasets/verify_dataset.py
```

### Training All Models
```bash
# Sequential training: M4 (EfficientNetV2-S) → M2 (Swin) → M3 (ConvNeXt)
python scripts/train_all.py

# Or individually:
python scripts/train.py --config configs/improved.yaml      # M4
python scripts/train.py --config configs/swin_tiny.yaml     # M2
python scripts/train.py --config configs/convnext_tiny.yaml # M3
```

### Generate Evidence
```bash
# Inference CSVs + Ensemble + McNemar test
python scripts/generate_evidence.py

# Figures (ROC, PR, confusion matrix) for all models
python scripts/generate_all_figures.py
```

## Dependencies

- **Deep Learning**: torch>=2.6.0, torchvision, timm>=1.0.0
- **Data**: numpy, pandas, pillow, opencv-python
- **Augmentation**: albumentations>=2.0.0
- **Explainability**: grad-cam>=1.5.0
- **Config**: pyyaml, tqdm
- **Dataset**: kaggle (for downloads)

## Experimental Results

Reproducible results from training on 9,224 dermatological images (80/10/10 split, seed=42).  
Hardware: NVIDIA RTX 5060 (8 GB), PyTorch 2.13+cu132, Windows 10.

| Model | Architecture | Accuracy | Train Time | Early Stop |
|---|---|---|---|---|
| **M2** | Swin Transformer Tiny | 86.62% | 2.98h | — |
| **M3** | ConvNeXt Tiny | **89.43%** | 0.93h | epoch 23 |
| **M4** | EfficientNetV2-S | 84.14% | 2.67h | epoch 41 |
| **M6** | Ensemble (M2+M3+M4) | 89.21% | — | — |

**McNemar test (M6 vs M4):** χ² = 25.49, p < 0.0001 — significant.

Per-sample predictions and full metrics are in `evidence/csvs/`.  
Complete report: `evidence/PathoVision_Relatorio_Completo.pdf` (8 pages).

## Citation

If you use this work, please cite:

```bibtex
@software{noronha2026pathovision,
  title={PathoVision v2: Multi-Architecture Deep Learning for Dermatological Image Classification},
  author={Noronha da Silva, Leandro and Roehrs, Alex and da Costa, Cristiano Andre and Lima, Kevin and Py, Monica Xavier and Moralles, Cassiano Ricardo Neubauer and da Costa, Luis Antonio L. F. and Schmidt, Douglas C.},
  year={2026},
  month={March},
  version={1.0.0.0},
  affiliation={Universidade do Vale do Rio dos Sinos (Unisinos)}
}
```
