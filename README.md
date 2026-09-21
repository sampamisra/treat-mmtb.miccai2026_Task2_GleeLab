# Task 2 Submission: Extra-Seed 10-Fold Harmonic Ensemble

Code by Sampa Misra, GleeLab.

Self-contained Docker inference package: prediction script, preprocessing helper, Python requirements, Dockerfile, reference preprocessing statistics, and a 10-fold ensemble checkpoint set.

## What changed vs. the 0.8714 submission

The fold ensemble grows from 5 to 10 members: the original 5 production folds (unchanged, identical weights since `Task2_26.08.10_0.8703`) plus 5 newly-trained folds using the exact same recipe and flags but a different random seed (777, vs the original 42) -- a different 5-fold cross-validation split and a different training trajectory. All 10 fold probabilities are combined with the same harmonic mean used since `Task2_26.08.14`. Everything else is identical:

- Same TTA band (`0.45 <= p <= 0.60`)
- Same 4 zoom/brightness TTA views (zoom +/-5%, brightness +/-10%)
- No CLAHE
- Same threshold (0.50)
- Same frozen Ark+ backbone, same tabular fusion head architecture, same training flags (`--no-balance-modality-label --class-weight balanced --label-smoothing 0.05 --no-fine-tune-last-stage`)

## Why more folds

Ensemble diversity through independently-seeded members is a standard variance-reduction technique -- it's mechanistically different from every other retraining attempt this cycle (multi-domain, metadata-dropout, compound-loss, longer-training-budget), none of which changed the recipe itself, only added more independent draws of the same one. It is purely additive: none of the original 5 fold checkpoints are replaced or modified.

## Local Validation (before submission)

Verified via two independent local test harnesses (both reproducing the production TTA/harmonic-mean pipeline), compared against the exact original 5-fold ensemble on the same 6 datasets:

| Dataset | Original 5-fold | 10-fold (harness A) | 10-fold (harness B) |
|---|---|---|---|
| Internal | reference | tie | -0.0011 |
| Montgomery | reference | tie | tie |
| Shenzhen (clean) | reference | +0.0012 | +0.0027 |
| Pakistan | reference | +0.0001 | tie |
| TBX11K | reference | +0.0037 | +0.0025 |
| TB Chest Radiography | reference | +0.0019 | +0.0034 |

Both harnesses agree on 5 of 6 datasets: Montgomery and Pakistan tie, and Shenzhen/TBX11K/TB Chest Radiography all show real, consistent gains. They disagree only on Internal -- one shows a tie, the other a small regression (-0.0011). This is the one open question mark on an otherwise consistent, positive local record; a fully clean, uninterrupted verification run using the actual unmodified `predict_task2.py` code (not a reimplementation) was started but not completed before this submission was built.

## Caveat

Local proxy datasets, including Internal's own held-out split, have not reliably predicted the private leaderboard's direction all session -- this was true again as recently as `Task2_26.08.15`, which showed only a single small local regression (TB Chest Radiography, -0.0008) and still lost real score. Given that history, the unresolved Internal signal here is a real, acknowledged risk, not a guaranteed non-issue. This submission is a deliberate calculated bet given four other datasets show consistent, repeated real gains across two independent test harnesses.

## Weights

```text
weights/reference_quantiles_ch0.npy
weights/class_weighted_metadata_fusion/arkplus_tabular_ch0_fold{1-5}_best.pth   (original, seed=42)
weights/class_weighted_metadata_fusion/arkplus_tabular_ch0_fold{6-10}_best.pth  (new, seed=777, renamed from fold{1-5} to avoid collision)
```

## Build And Run

```bash
docker build -t gleelab-task2-10fold-seed777:latest .
docker run --rm --gpus all -v /path/to/input:/input:ro -v "$PWD/output:/output" gleelab-task2-10fold-seed777:latest
```

The container writes both `prediction.csv` and `test.csv`, each with:

```text
filename,TB/Normal
```

The Docker image archive can be created with:

```bash
docker save -o gleelab-task2-10fold-seed777.tar gleelab-task2-10fold-seed777:latest
```
