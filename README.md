# TREAT-MMTB 2026 — Task 2: Class-Weighted Metadata-Aware Ark+ Fusion

## Team GleeLab 
TB / Normal classification from frontal chest radiographs and structured clinical metadata (age, sex). Private leaderboard F1 = **0.8727** (Rank 1, Task 2).

This repository contains the full inference pipeline (Docker), the training code, and the trained model weights.

## Dataset

- **Training set:** 7,757 frontal chest radiographs (TB / Normal), with age, sex, and imaging modality (CR, DX, XA, XC) available for each case.
- **Internal validation set:** 1,940 radiographs, same format.
- **Supplementary external evaluation** (never used for training): Montgomery, Shenzhen, Pakistan TB, and TBX11K — 8,408 images total. Age/sex metadata is available for Montgomery and Shenzhen; not available for Pakistan TB or TBX11K.
- Imaging modality was excluded from the model input entirely: it was highly imbalanced by diagnostic label in the training data (some modalities were almost exclusively TB or almost exclusively Normal), which risked the model learning a modality-label shortcut instead of genuine radiographic findings. Only age and sex are used as auxiliary metadata.

## Preprocessing

Applied identically at training, validation, and inference time (`border_marker_cleanup.py` + preprocessing routines in `predict_task2.py`):

1. Conservative border/corner artifact removal (black borders, corner/edge markers, opaque marks, label blocks — common in this dataset and not indicative of TB).
2. Reference quantile intensity matching, harmonizing each image's intensity distribution against a fixed reference (`weights/reference_quantiles_ch0.npy`) to reduce cross-scanner/cross-site appearance differences.
3. Per-image z-score normalization.
4. Robust 0.5–99.5 percentile rescaling.
5. Resize to 224×224, replicate to 3 channels, standard ImageNet normalization.

## Model Architecture

- **Image encoder:** Ark+ Swin-Base224, a chest-radiograph foundation model, used as a **frozen** backbone (no backbone weights are updated during training).
- **Metadata branch:** age (z-normalized using training-set statistics, mean=49.05, std=15.55) and sex (binary-encoded), passed through a small 2→64 fully-connected layer with ReLU. Missing age/sex values fall back to neutral defaults (training-set mean age, 0.5 midpoint for sex) rather than excluding the case.
- **Fusion:** the image feature vector and the metadata feature vector are concatenated and passed through a linear classification head (TB vs. Normal).
- Only the metadata branch and the classification head are trained; the Ark+ backbone is frozen throughout.

## Training Procedure & Parameters

- **Loss:** class-weighted cross-entropy with label smoothing (0.05), to address residual class imbalance without relying on modality-derived shortcuts.
- **Cross-validation:** two independently-seeded 5-fold splits (seed=42 and seed=777) of identical architecture and training configuration, giving a 10-model ensemble in total. The second seed group is purely additive — it does not replace or modify the first.
- **Key training flags:** `--no-balance-modality-label --class-weight balanced --label-smoothing 0.05 --no-fine-tune-last-stage`
- Training script: `training_code/train_task2_arkplus_fold_ensemble.py`.

## Testing / Evaluation

Fold probabilities are combined via the **harmonic mean** (not a simple average) — a deliberately conservative combination rule that only yields a high TB probability when the fold ensemble agrees with reasonable confidence. A fixed decision threshold of 0.50 is used; thresholds tuned on internal validation data did not transfer reliably to the external cohorts, so no per-dataset threshold tuning is applied.

| Dataset | F1 |
|---|---|
| Internal validation | 0.9901 |
| Montgomery (external) | 0.9381 |
| Shenzhen (external) | 0.9401 |
| Pakistan TB (external) | 0.9073 |
| TBX11K (external) | 0.8041 |
| **Private leaderboard (real, organizer-evaluated)** | **0.8727** |

## Post-processing

A selective test-time augmentation step re-examines only borderline/uncertain predictions using a small set of additional zoom and brightness views of the same image, then averages the result with the original prediction. Predictions the model is already confident about are left unchanged. Horizontal flip is deliberately not used, since chest radiograph anatomy is not left-right symmetric.

## Weights

```text
weights/reference_quantiles_ch0.npy
weights/class_weighted_metadata_fusion/arkplus_tabular_ch0_fold{1-5}_best.pth   (seed=42)
weights/class_weighted_metadata_fusion/arkplus_tabular_ch0_fold{6-10}_best.pth  (seed=777)
```

## Build and Run

```bash
docker build -t gleelab-task2:latest .
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v "$PWD/output:/output" \
  gleelab-task2:latest
```

The container reads PNG images (and an optional metadata CSV) from `/input` and writes `prediction.csv` (and `test.csv`) to `/output`, each with columns:

```text
filename,TB/Normal
```

## Citation

Paper accepted to the TREAT-MMTB 2026 proceedings (Springer LNCS), presented at MICCAI 2026, Strasbourg, France.
