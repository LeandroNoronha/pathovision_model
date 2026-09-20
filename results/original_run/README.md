# Original-run results

Output files of the training and evaluation runs reported in the paper (second PathoVision iteration, cluster runs with seed 42). The per-model internal results (Table 6, rows M1 to M4b), the M1 cross-validation (Table 12), the external accuracies (Table 7), the skin-tone stratification (Table 10) and the dataset audit (Section 5.2) can be traced to these files; the hybrid (H1), ensemble (M6), M4 cross-validation, external per-class and efficiency figures come from run outputs that were not retained as files. The independent re-execution on consumer hardware is documented separately in `reproduction/`.

| Folder | Contents | Paper |
|---|---|---|
| `01_dataset_qc/` | dHash duplicate audit of the merged 13,298-image corpus: `dataset_qc_report.json` (summary and per-class brightness statistics) and `duplicate_pairs.txt` (every duplicate pair found, with the inherited partition of each copy) | Sections 4.3.2 and 5.2 |
| `02_baseline/` | M1, EfficientNetB2, cross-entropy, unbalanced | Table 6 |
| `02b_baseline_balanced/` | M1b, EfficientNetB2, cross-entropy + balanced sampling | Table 6 |
| `05_focal_balanced/` | M5, EfficientNetB2, focal loss + balanced sampling | Table 6 |
| `03_m2_v2s_balanced/`, `03_m2b_v2s_nobal/` | M2 / M2b, EfficientNetV2-S | Table 6 |
| `03_m3_swin_balanced/`, `03_m3b_swin_nobal/` | M3 / M3b, Swin-Tiny | Table 6 |
| `03_m4_convnext_balanced/`, `03_m4b_convnext_nobal/` | M4 / M4b, ConvNeXt-Tiny. The confusion matrix, ROC and PR curves of M4 are the paper's Figures 3 to 5 | Table 6, Figures 3 to 5 |
| `04_cross_validation/` | Five-fold stratified cross-validation of the baseline M1 (mean, standard deviation and 95% CI per metric) | Table 12 |
| `05_ablation/` | Best-validation accuracy, loss and epoch of every training run (one row per model, plus the five M1 folds); the source of the convergence epochs quoted in Section 4.6. The `ce_loss_only_(no_weights)` row and the table caption come from a preliminary single-factor run that is not reported in the paper | Section 4.6 |
| `07_fairness/` | Image-level ITA estimates of the 929 test images (`skin_tone_annotations.csv`), the accuracy per stratum (`fairness_metrics.csv`) and the corresponding plots | Table 10, Section 5.8 |
| `08_external_validation/` | Internal versus external accuracy and weighted F1 of all nine end-to-end models on the 4,804-image external set (`unified_external_validation.csv`) | Table 7 |

Each model folder contains the normalized and absolute confusion matrices, the ROC and PR curves, the per-class metrics (`*_per_class.csv`), a LaTeX table and a `*_comparison.csv` that lists the per-class F1 next to the first-iteration paper values.

Notes. The per-class CSVs are the authoritative source of the internal metrics; the `Int Acc`/`Int F1` columns of `08_external_validation/unified_external_validation.csv` were copied from an interim summary and differ for M2 (86.65 vs. 83.85 from the per-class file) and M3 (85.47 vs. 85.36). The ITA values in `07_fairness/` are on the 8-bit OpenCV CIELab scale described in the paper (Section 4.13); the stratum labels are nominal. Per-sample prediction files of these runs were not retained, which is why the paper reports paired tests only for the re-execution in `reproduction/`. Grad-CAM panels of individual test photographs are not redistributed here because they contain the source images; the four examples shown in the paper are reproduced under the source repositories' research-use terms.
