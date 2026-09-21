# Task 2 Submission: 10-Fold Harmonic Ensemble, Wider TTA Band (0.45-0.62)

Code by Sampa Misra, Glee lab.

Self-contained Docker inference package: prediction script, preprocessing helper, Python requirements, Dockerfile, reference preprocessing statistics, and the same 10-fold ensemble checkpoint set shipped in `Task2_26.08.16_0.8727`.

## What changed vs. the 0.8727 submission

Exactly one constant, in `predict_task2.py`:

```python
TTA_LOW = 0.45
TTA_HIGH = 0.60   # previous value in Task2_26.08.16_0.8727
```

changed to:

```python
TTA_LOW = 0.45
TTA_HIGH = 0.62
```

Nothing else differs -- same 10-fold checkpoint set (5 original seed=42 folds + 5 seed=777 folds), same harmonic-mean fold combination, same 4 zoom/brightness TTA views, same threshold (0.50), no retraining.

## Why this change

The narrow-band TTA re-examines any prediction whose base probability falls in `[TTA_LOW, TTA_HIGH]`. That band was originally tuned (`Task2_26.08.12`) against the 5-fold ensemble's probability distribution. Adding the seed=777 folds on `Task2_26.08.16` changed the ensemble's output distribution, so the old band width was never re-validated against the actual 10-fold ensemble until now.

## Local Validation (before submission)

Re-swept `TTA_HIGH` on the real, unmodified `predict_task2.py` (imported directly, not reimplemented) against the exact shipped 10-fold ensemble, across all 6 local datasets. `TTA_LOW=0.45` held fixed (known hard boundary from earlier band re-sweeps -- going below it reproduces a real Shenzhen regression).

| Dataset | 0.45-0.60 (shipped, 0.8727) | 0.45-0.62 | 0.45-0.65 | 0.45-0.68 |
|---|---|---|---|---|
| Internal | 0.9901 | 0.9901 | 0.9901 | 0.9901 |
| Montgomery | 0.9381 | 0.9381 | 0.9381 | 0.9381 |
| Shenzhen (clean) | 0.9401 | 0.9401 | 0.9401 | 0.9401 |
| Pakistan | 0.9073 | 0.9073 | 0.9073 | 0.9073 |
| TBX11K | 0.8041 | **0.8045** | 0.8050 | 0.8050 |
| TB Chest Radiography | 0.5915 | 0.5915 | 0.5915 | 0.5915 |

Every dataset except TBX11K is completely flat across the entire grid -- zero change at any band width tested, including Internal (unlike `Task2_26.08.16`, which shipped with one small Internal-only regression). TBX11K climbs monotonically with band width and plateaus at 0.65 (0.68 adds nothing further). The 0.62 band keeps the change smaller while preserving a measurable TBX11K gain and no observed local regressions.

## Why 0.62

0.62 is the smallest tested widening that improves TBX11K without changing any other local dataset. This keeps the package close to the previously-shipped, organizer-verified recipe while avoiding the wider 0.65 band.

## Weights

```text
weights/reference_quantiles_ch0.npy
weights/class_weighted_metadata_fusion/arkplus_tabular_ch0_fold{1-5}_best.pth   (original, seed=42)
weights/class_weighted_metadata_fusion/arkplus_tabular_ch0_fold{6-10}_best.pth  (seed=777, renamed from fold{1-5} to avoid collision)
```

Identical to `Task2_26.08.16_0.8727` -- weights are untouched, only the TTA band constant changed.

## Build And Run

```bash
docker build -t gleelab-task2-band062:latest .
docker run --rm --gpus all -v /path/to/input:/input:ro -v "$PWD/output:/output" gleelab-task2-band062:latest
```

The container writes both `prediction.csv` and `test.csv`, each with:

```text
filename,TB/Normal
```

Smoke-tested locally against 2 sample images (`experiments/smoke_safe_tta_input/`), correct output format confirmed.

The Docker image archive can be created with:

```bash
docker save -o gleelab-task2-band062.tar gleelab-task2-band062:latest
```
