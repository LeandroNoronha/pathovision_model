# PathoVision: Multi-Architecture Deep Learning for Dermatological Image Classification

This repository contains the complete implementation and research materials for the PathoVision article (second iteration of the framework).
The project develops a leakage-aware, externally validated benchmark for automated classification of seven common non-neoplastic dermatological conditions from clinical photographs, with emphasis on data curation, class-balancing analysis, external validation, explainability, and an exploratory skin-tone fairness evaluation.

## Versioning

The evolution of this project has generated different versions of the classifier:

- **v1.0.0.0 (Stable)**
  - PathoVision v2: complete rewrite of the pipeline
  - Multi-architecture comparison (EfficientNetB2 baseline, EfficientNetV2-S, Swin-Tiny, ConvNeXt-Tiny) under balanced and unbalanced training
  - Dataset cleaning pipeline (9,227 images after perceptual-hash deduplication)
  - External validation on 4,804 independent images (four Kaggle sources + SD-198)
  - Explainability with Grad-CAM analysis
  - Fairness evaluation with ITA skin-tone stratification
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

PathoVision v2 compares one legacy baseline (EfficientNetB2) and three modern backbones (EfficientNetV2-S, Swin-Tiny, ConvNeXt-Tiny), each trained with and without class balancing (balanced sampling and focal loss), plus a hybrid random-forest configuration on ConvNeXt embeddings and a soft-voting ensemble.

**Key Features:**
- **Dataset Pipeline**: automated download, class mapping, perceptual-hash (dHash) duplicate and leakage detection, stratified splitting, and validation
- **Multi-Architecture**: EfficientNetB2, EfficientNetV2-S, Swin-Tiny, ConvNeXt-Tiny, hybrid CNN + random forest, and a soft-voting ensemble
- **External Validation**: zero-shot evaluation on 4,804 images from five independent public sources
- **Explainability**: Grad-CAM analysis for model interpretability
- **Fairness Evaluation**: image-level ITA skin-tone stratification
- **Testing**: test suite with pytest
- **Tooling**: Ruff linting/formatting, type hints, reproducible YAML configurations

**Internal Dataset:**
- Sources: Human Skin Diseases (Kaggle) + DermNet (Kaggle)
- Classes: 7 dermatological conditions (Acne, Candidiasis, Eczema, NailFungus, Normal, Psoriasis, Tinea)
- Final dataset: 9,227 images (7,378 train / 920 val / 929 test), stratified 80/10/10 split, seed 42
- Duplicate removal: dHash (difference hash, perceptual) deduplication removed exact and near-duplicate images and 1,017 train–test leakage groups

**External Validation Dataset:**
- 4,804 images from four Kaggle repositories (Acne Dataset; Nail Disease Image Classification; Skin Disease and Normal Skin; Skin Diseases Image Dataset) plus SD-198 (HuggingFace)
- Source labels mapped to the seven classes at label granularity; unmapped labels discarded and recorded
- Deduplicated against the internal corpus with the same dHash procedure
- Label-level curation metadata in `datasets/external_unified/` (see below)

## Repository Structure

- `src/` - Core library code (data, models, training, evaluation)
- `scripts/` - Experiment scripts (training, evaluation, cross-validation, external validation, Grad-CAM, fairness, efficiency, hybrid model)
- `configs/` - YAML training configurations for every model of the paper
- `datasets/` - Dataset download, organization, deduplication, external-set construction, and verification tools
- `datasets/external_unified/` - Label-level curation metadata of the external validation set (`external_label_mapping.csv`, `external_discarded_labels.csv`, `external_manifest.csv`)
- `tests/` - Test suite with pytest
- `reproduction/` - Independent re-execution of the training pipeline (see "Independent reproduction" below): its own scripts, configs and evidence package (`reproduction/evidence/`)

## Model identifiers

Model IDs follow the paper. The table maps each ID to its training configuration.

| ID | Backbone | Balancing | Config |
|---|---|---|---|
| M1 | EfficientNetB2 | none (legacy baseline) | `configs/base.yaml` |
| M1b | EfficientNetB2 | cross-entropy + balanced sampling | `configs/balanced_ce.yaml` |
| M5 | EfficientNetB2 | focal loss + balanced sampling | `configs/balanced.yaml` |
| M2b | EfficientNetV2-S | none | `configs/improved_nobal.yaml` |
| M2 | EfficientNetV2-S | focal loss + balanced sampling | `configs/improved.yaml` |
| M3b | Swin-Tiny | none | `configs/swin_tiny_nobal.yaml` |
| M3 | Swin-Tiny | focal loss + balanced sampling | `configs/swin_tiny.yaml` |
| M4b | ConvNeXt-Tiny | none | `configs/convnext_tiny_nobal.yaml` |
| M4 | ConvNeXt-Tiny | focal loss + balanced sampling | `configs/convnext_tiny.yaml` |
| H1 | Random forest on M4 embeddings | – | `scripts/train_hybrid.py` |
| M6 | Soft-voting ensemble of M2 + M3 + M4 | – | `scripts/evaluate.py` |

`configs/dinov2_vit.yaml` is an exploratory configuration not used in the paper.

## Quick Start

### Prerequisites
- Python 3.13+
- PyTorch 2.6+
- CUDA 12.4+ (recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Internal dataset preparation
```bash
# Download datasets from Kaggle (requires kaggle.json token)
python datasets/download_kaggle.py

# Organize and merge classes
python datasets/organize_dataset.py

# Detect and remove duplicates (dHash perceptual)
python datasets/detect_duplicates2.py --dataset-dir datasets/merged --method dhash --delete

# Verify final dataset
python datasets/verify_dataset.py
```

### Training
```bash
python scripts/train.py --config configs/convnext_tiny.yaml   # M4 (primary model)
python scripts/train.py --config configs/swin_tiny.yaml       # M3
python scripts/train.py --config configs/improved.yaml        # M2
python scripts/train.py --config configs/base.yaml            # M1 (baseline)
```

### Evaluation, cross-validation and analyses
```bash
python scripts/evaluate.py            # test-set metrics and ensemble
python scripts/cross_validate.py      # five-fold stratified cross-validation
python scripts/fairness_analysis.py   # ITA skin-tone stratification
python scripts/gradcam_demo.py        # Grad-CAM
python scripts/efficiency_benchmark.py
```

### External validation set
```bash
# Build the unified external set (4 Kaggle sources + SD-198) and write the
# label-level curation metadata to datasets/external_unified/.
# Run once with --dry-run to review the SD-198 label selection.
python datasets/build_external_unified.py \
    --internal-dir datasets/merged datasets/final \
    --raw-dir datasets/external_raw \
    --out-dir datasets/external_unified

# Zero-shot evaluation of a trained model on the external set
python scripts/external_validation_unified.py --checkpoint <path/to/best.pt>
```

Note: the public sources change upstream over time, so re-executing the builder reproduces the label-level mapping rules and discard records rather than guaranteeing the exact image-level composition evaluated in the paper.

## Dependencies

- **Deep Learning**: torch>=2.6.0, torchvision, timm>=1.0.0
- **Data**: numpy, pandas, pillow, opencv-python
- **Augmentation**: albumentations>=2.0.0
- **Explainability**: grad-cam>=1.5.0
- **Config**: pyyaml, tqdm
- **Datasets**: kaggle (Kaggle downloads), datasets (HuggingFace, SD-198)

## Experimental Results (paper)

Held-out internal test set of 929 images (stratified 80/10/10 split of the 9,227-image corpus, seed 42). Full per-class results, confidence intervals, cross-validation, external validation and fairness analyses are reported in the paper.

| ID | Model | Accuracy (%) | Weighted F1 (%) |
|---|---|---|---|
| M1 | EfficientNetB2 (baseline) | 81.16 | 81.07 |
| M1b | EfficientNetB2 + CE + balanced | 85.36 | 85.31 |
| M5 | EfficientNetB2 + focal + balanced | 82.45 | 82.28 |
| M2b | EfficientNetV2-S | 84.39 | 84.29 |
| M2 | EfficientNetV2-S + focal + balanced | 86.65 | 86.57 |
| M3b | Swin-Tiny | 86.22 | 86.18 |
| M3 | Swin-Tiny + focal + balanced | 85.36 | 85.31 |
| M4b | ConvNeXt-Tiny | 87.73 | 87.68 |
| **M4** | **ConvNeXt-Tiny + focal + balanced (primary model)** | **87.84** | **87.73** |
| H1 | Random forest on M4 embeddings | 87.51 | 87.42 |
| M6 | Soft-voting ensemble (M2 + M3 + M4), exploratory | 89.45 | 89.39 |

- Five-fold cross-validation: M4 86.65 ± 0.80% vs. M1 80.85 ± 0.80% (paired t-test p = 0.0002).
- External validation (4,804 images): M4 66.94% (internal 87.84%, drop 20.9 pp); M1 55.41%.
- The ensemble's +1.61 pp margin over M4 is not statistically confirmed (overlapping confidence intervals); see the independent reproduction below.

## Independent reproduction (`reproduction/`)

The full training pipeline (dataset download, dHash deduplication, stratified split, training of all backbones from scratch) was independently re-executed by a co-author on separate hardware (NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, PyTorch 2.13+cu132, Windows 10, 30–31 July 2026). The scripts, configs and the complete evidence package of that run (per-sample prediction CSVs, figures, McNemar test, PDF report) are in `reproduction/`.

Owing to dataset-version and stochastic differences, the re-execution retained 9,224 images (927 test) and yields somewhat different accuracies from the paper.

**Important: model IDs inside `reproduction/` follow a different convention from the paper.**

| `reproduction/` ID | Backbone | Paper ID | Re-execution accuracy | Training time |
|---|---|---|---|---|
| M2 | Swin-Tiny | M3 | 86.62% | 2.98 h |
| M3 | ConvNeXt-Tiny | M4 | 89.43% | 0.93 h |
| M4 | EfficientNetV2-S | M2 | 84.14% | 2.67 h |
| M4bal | EfficientNetB2 (balanced) | M1b / M5 | 80.04% | 7.25 h |
| M6 | Ensemble (Swin + ConvNeXt + EfficientNetV2-S) | M6 | 89.21% | – |

Key finding of the reproduction: the re-trained ConvNeXt-Tiny (89.43%) marginally outperformed the re-trained ensemble (89.21%). A paired McNemar test on the shared per-sample predictions (927 images; 13 vs. 11 discordant pairs) found no significant difference between them (continuity-corrected χ² = 0.04, p = 0.84), which is the result reported in the paper. The `reproduction/evidence/reports/mcnemar_test.txt` file additionally reports the ensemble against the re-trained EfficientNetV2-S (χ² = 25.49, p < 0.0001); that comparison is not the one discussed in the paper.

```bash
# Re-run the reproduction pipeline
python reproduction/scripts/train_all.py
python reproduction/scripts/generate_evidence.py       # inference CSVs, ensemble, McNemar
python reproduction/scripts/generate_all_figures.py    # ROC, PR, confusion matrices
```

## Citation

If you use this work, please cite:

```bibtex
@software{noronha2026pathovision,
  title={PathoVision v2: Multi-Architecture Deep Learning for Dermatological Image Classification},
  author={Noronha da Silva, Leandro and Roehrs, Alex and da Costa, Cristiano Andre and Lima, Kevin and Py, Monica Xavier and Moralles, Cassiano Ricardo Neubauer and da Costa, Luis Antonio L. F. and Rigo, Sandro J. and Schmidt, Douglas C. and Zhou, Gang},
  year={2026},
  version={1.0.0.0},
  affiliation={Universidade do Vale do Rio dos Sinos (Unisinos)}
}
```
