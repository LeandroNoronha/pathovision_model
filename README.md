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
- Duplicate Removal: Perceptual-hash deduplication removed exact and near-duplicate images and 1,017 train–test leakage groups

**Architectures Supported:**
- ViT (DINOv2) - Vision Transformer
- Swin Transformer - Hierarchical vision transformer
- ConvNeXt - Modern convolutional network
- Hybrid Ensemble - Feature extraction + classical ML

## Repository Structure

- `src/` - Core library code (data, models, training, evaluation)
- `scripts/` - Reproducible experiments (train.py, evaluate.py, gradcam_demo.py)
- `configs/` - YAML training configurations (17 config files for different architectures)
- `datasets/` - Dataset organization and cleaning tools
- `tests/` - Test suite with pytest configuration

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
# Download datasets from Kaggle
python datasets/download_kaggle.py

# Organize and clean dataset
python datasets/organize_dataset.py

# Detect and remove duplicates
python datasets/detect_duplicates.py
python datasets/detect_duplicates2.py

# Verify final dataset
python datasets/verify_dataset.py
```

### Basic Usage
```bash
# Train a model
python scripts/train.py --config configs/balanced.yaml

# Generate Grad-CAM explanations
python scripts/gradcam_demo.py --model-path outputs/best_model.pt

# Evaluate fairness
python scripts/fairness_analysis.py --model-path outputs/best_model.pt

# Run ablation studies
python scripts/ablation_study.py
```

## Dependencies

- **Deep Learning**: torch>=2.6.0, torchvision, timm>=1.0.0
- **Data**: numpy, pandas, pillow, opencv-python
- **Augmentation**: albumentations>=2.0.0
- **Explainability**: grad-cam>=1.5.0
- **Config**: pyyaml, tqdm
- **Dataset**: kaggle (for downloads)

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
