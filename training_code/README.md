# Training code for Task2_26.08.16_0.8727 (real private F1 = 0.8727, current best)

This submission's 10-fold ensemble (`weights/class_weighted_metadata_fusion/fold1`-`fold10`)
was produced by training the SAME recipe twice with two different random seeds, additively
combined (not a replacement of one split with another).

`train_task2_arkplus_fold_ensemble.py` is the parameterized training script (seed and
output directory are CLI flags). 

## fold1-fold5 (original, seed=42)

```
python train_task2_arkplus_fold_ensemble.py \
  --seed 42 \
  --output-dir <original weights dir> \
  --no-balance-modality-label \
  --class-weight balanced \
  --label-smoothing 0.05 \
  --no-fine-tune-last-stage
```

Production copy of these 5 checkpoints lives at
`models/class_weighted_metadata_fusion/weights/` in the project root 
## fold6-fold10 (extra diversity, seed=777, renamed fold1-5 -> fold6-10 to avoid collision)

```
python train_task2_arkplus_fold_ensemble.py \
  --seed 777 \
  --output-dir experiments/extra_seed_folds/weights \
  --no-balance-modality-label \
  --class-weight balanced \
  --label-smoothing 0.05 \
  --no-fine-tune-last-stage
```

Flags confirmed identical to the seed=42 group via checkpoint metadata inspection
(`balance_modality_label=False`, `class_weight=balanced`, `label_smoothing=0.05`,
`fine_tune_last_stage=False`, `tab_hidden=64`) -- same recipe, only the random seed differs,
so the two fold groups are independently-initialized/independently-split members of the same
ensemble family, not a different method.

## Why this combination, not a straight 10-fold CV

A separate experiment that used `--k-folds 10` (replacing the 5-fold split with a 10-fold
split under the same seed=42, instead of adding a second independently-seeded 5-fold group)
tested WORSE locally (regressed Internal AND Shenzhen) than this additive approach (which
only ever regressed Internal, and even that was outweighed by real-world gains on every
external proxy dataset). See `submission_history.md` for the full comparison table.

Both fold groups are combined at inference time via harmonic-mean fold ensembling
(`1.0 / (1.0/fold_probs.clamp_min(1e-8)).mean(dim=0)`) exactly as `predict_task2.py` does for
any N-fold ensemble -- no special-casing between the two seed groups.
