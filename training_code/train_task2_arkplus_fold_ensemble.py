"""
Ark+ Swin-Base224 + tabular fusion (age, gender) for Task 2 TB/Normal classification.
Modality_DICOM is deliberately EXCLUDED from the tabular branch: it is a
near-deterministic shortcut in this dataset (XA/XC are ~100% TB, DX is ~99%
Normal) and including it would reproduce the same shortcut-inflation failure
already observed when unfreezing more of the backbone.

Backbone stays fully frozen throughout (this is what generalized best in prior
experiments); only the tabular MLP + fusion classifier head are trained.

By Sampa Misra
"""

import argparse
import gc
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from arkplus_base224_task2_model import build_arkplus_swin_base224, load_arkplus_base224_weights
from train_task2_arkplus_bias_mitigated import (
    LABEL_MAP,
    set_seed,
    image_to_rgb,
    build_transform,
    count_pngs,
    print_modality_table,
    filter_by_modality,
    build_reference_quantiles,
    metrics_from_predictions,
    find_best_threshold,
    split_cr_calibration,
    build_modality_label_sampler,
    print_metric_line,
    print_per_modality_metrics,
    print_cv_summary,
    BiasMitigatedTBNormalDataset,
    load_checkpoint_model_and_threshold as load_image_checkpoint_model_and_threshold,
)

NUM_CLASSES = 2


class TabularFusionDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        preprocess_type,
        transform,
        age_mean,
        age_std,
        artifact_clean=True,
        reference_quantiles=None,
        return_meta=False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.preprocess_type = preprocess_type
        self.transform = transform
        self.age_mean = age_mean
        self.age_std = age_std
        self.artifact_clean = artifact_clean
        self.reference_quantiles = reference_quantiles
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = image_to_rgb(
            self.img_dir / f"{row['new_id']}.png",
            self.preprocess_type,
            artifact_clean=self.artifact_clean,
            reference_quantiles=self.reference_quantiles,
        )
        label = LABEL_MAP[str(row["TB/Normal"]).strip().lower()]
        if self.transform is not None:
            image = self.transform(image)

        age_norm = (float(row["age"]) - self.age_mean) / (self.age_std + 1e-8)
        gender_bin = 1.0 if str(row["gender"]).strip().lower() == "male" else 0.0
        tabular = torch.tensor([age_norm, gender_bin], dtype=torch.float32)

        if self.return_meta:
            modality = str(row.get("Modality_DICOM", "NA")).upper()
            return image, tabular, torch.tensor(label, dtype=torch.long), modality
        return image, tabular, torch.tensor(label, dtype=torch.long)


class TabularFusionModel(nn.Module):
    def __init__(self, backbone, tabular_dim=2, num_classes=2, tab_hidden=16):
        super().__init__()
        self.backbone = backbone
        if hasattr(backbone.head, "fc"):
            img_feat_dim = backbone.head.fc.in_features
        else:
            img_feat_dim = backbone.head.in_features
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tab_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(img_feat_dim + tab_hidden, num_classes)

    def forward(self, image, tabular):
        feats = self.backbone.forward_features(image)
        if hasattr(self.backbone, "forward_head"):
            pooled = self.backbone.forward_head(feats, pre_logits=True)
        else:
            pooled = feats
        tab = self.tabular_mlp(tabular)
        fused = torch.cat([pooled, tab], dim=1)
        return self.classifier(fused)


def split_threshold_calibration(labels, probs, modalities, source):
    if source == "all":
        return np.asarray(labels), np.asarray(probs), "all-val"
    return split_cr_calibration(labels, probs, modalities)


def build_class_weight(df, device):
    labels = df["TB/Normal"].astype(str).str.strip().str.lower().map(LABEL_MAP).to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def unfreeze_last_swin_stage(backbone):
    for name, param in backbone.named_parameters():
        if (
            name.startswith("layers.3.")
            or name.startswith("norm.")
        ):
            param.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Fine-tuning last Swin stage. Backbone trainable parameters: {trainable:,} / {total:,}")
    return backbone


def create_model(args, device):
    backbone = build_arkplus_swin_base224(num_classes=NUM_CLASSES)
    backbone = load_arkplus_base224_weights(backbone, args.ark_checkpoint)
    for p in backbone.parameters():
        p.requires_grad = False
    if args.fine_tune_last_stage:
        backbone = unfreeze_last_swin_stage(backbone)
    model = TabularFusionModel(
        backbone,
        tabular_dim=2,
        num_classes=NUM_CLASSES,
        tab_hidden=args.tab_hidden,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, n = 0.0, 0
    for images, tabular, labels in loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, tabular)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        n += images.size(0)
    return loss_sum / n if n else 0.0


@torch.no_grad()
def collect_predictions(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    labels_all, probs_all, modalities_all = [], [], []
    for batch in loader:
        if len(batch) == 4:
            images, tabular, labels, modalities = batch
            modalities_all.extend(list(modalities))
        else:
            images, tabular, labels = batch
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images, tabular)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        loss_sum += loss.item() * images.size(0)
        n += images.size(0)
        labels_all.extend(labels.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
    return loss_sum / n if n else 0.0, labels_all, probs_all, modalities_all


def run_fold(fold, train_df, val_df, transform, age_mean, age_std, reference_quantiles, args, device):
    print(f"\n========== Ark+ Tabular-Fusion Fold {fold}/{args.k_folds} ==========")
    print(f"Training images   : {len(train_df)}")
    print(f"Validation images : {len(val_df)}")
    if args.balance_modality_label:
        print("Training sampler  : balanced by Modality_DICOM x TB/Normal")

    train_ds = TabularFusionDataset(
        train_df, args.train_img_dir, args.preprocess_type, transform,
        age_mean, age_std, artifact_clean=args.artifact_clean,
        reference_quantiles=reference_quantiles,
    )
    val_ds = TabularFusionDataset(
        val_df, args.train_img_dir, args.preprocess_type, transform,
        age_mean, age_std, artifact_clean=args.artifact_clean,
        reference_quantiles=reference_quantiles, return_meta=True,
    )
    sampler = build_modality_label_sampler(train_df) if args.balance_modality_label else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None,
                               sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = create_model(args, device)
    class_weight = build_class_weight(train_df, device) if args.class_weight == "balanced" else None
    if class_weight is not None:
        print(f"Class loss weight : normal={class_weight[0].item():.3f} tb={class_weight[1].item():.3f}")
    criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / f"arkplus_tabular_{args.preprocess_type}_fold{fold}_best.pth"
    last_path = args.output_dir / f"arkplus_tabular_{args.preprocess_type}_fold{fold}_last.pth"

    best_f1 = -1.0
    best_metrics = None
    best_epoch = 0
    stale_epochs = 0
    swa_state_queue = deque(maxlen=args.swa_last_n_epochs) if args.swa else None

    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, labels, probs, modalities = collect_predictions(model, val_loader, criterion, device)

        if swa_state_queue is not None:
            # Only the trainable head (backbone is frozen, so its weights never move --
            # no point storing/averaging them). Cloned to CPU so the queue doesn't pin
            # GPU memory for `swa_last_n_epochs` copies of the head.
            trainable_state = {
                k: v.detach().clone().cpu()
                for k, v in model.state_dict().items()
                if k.startswith("tabular_mlp.") or k.startswith("classifier.")
            }
            swa_state_queue.append(trainable_state)

        default_metrics = metrics_from_predictions(
            labels, (np.asarray(probs) >= 0.5).astype(np.int64), probs,
        )
        threshold_labels, threshold_probs, threshold_source = split_threshold_calibration(
            labels, probs, modalities, args.threshold_calibration,
        )
        threshold, threshold_metrics = find_best_threshold(threshold_labels, threshold_probs)
        full_val_metrics_at_threshold = metrics_from_predictions(
            labels, (np.asarray(probs) >= threshold).astype(np.int64), probs,
        )
        selection_f1 = (
            full_val_metrics_at_threshold["f1"]
            if args.checkpoint_metric == "full-val-f1"
            else threshold_metrics["f1"]
        )

        if selection_f1 > best_f1:
            best_f1 = selection_f1
            best_metrics = full_val_metrics_at_threshold.copy()
            best_metrics["threshold"] = threshold
            best_metrics["threshold_source"] = threshold_source
            best_metrics["threshold_source_f1"] = threshold_metrics["f1"]
            best_metrics["checkpoint_metric"] = args.checkpoint_metric
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "threshold_source": threshold_source,
                    "metrics": best_metrics,
                    "epoch": best_epoch,
                    "age_mean": age_mean,
                    "age_std": age_std,
                    "tab_hidden": args.tab_hidden,
                    "balance_modality_label": args.balance_modality_label,
                    "class_weight": args.class_weight,
                    "label_smoothing": args.label_smoothing,
                    "threshold_calibration": args.threshold_calibration,
                    "checkpoint_metric": args.checkpoint_metric,
                    "fine_tune_last_stage": args.fine_tune_last_stage,
                },
                best_path,
            )
        else:
            stale_epochs += 1

        print(
            f"Fold [{fold}/{args.k_folds}] Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            f"Default Val F1: {default_metrics['f1']:.4f} "
            f"{threshold_source} F1: {threshold_metrics['f1']:.4f} "
            f"Full Val F1@Thr: {full_val_metrics_at_threshold['f1']:.4f} "
            f"Best {args.checkpoint_metric}: {best_f1:.4f} "
            f"Time: {time.time() - start:.2f}s"
        )

        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; best {args.checkpoint_metric}={best_f1:.4f} at epoch {best_epoch}.")
            break

    torch.save(model.state_dict(), last_path)
    print(f"Saved best model to: {best_path}")
    print(f"Saved last model to: {last_path}")

    if swa_state_queue is not None and len(swa_state_queue) > 0:
        avg_state = {
            key: torch.stack([s[key].float() for s in swa_state_queue], dim=0).mean(dim=0)
            for key in swa_state_queue[0]
        }
        swa_model = create_model(args, device)
        swa_full_state = swa_model.state_dict()
        swa_full_state.update(avg_state)
        swa_model.load_state_dict(swa_full_state)
        swa_model.eval()

        swa_val_loss, swa_labels, swa_probs, swa_modalities = collect_predictions(swa_model, val_loader, criterion, device)
        swa_threshold_labels, swa_threshold_probs, swa_threshold_source = split_threshold_calibration(
            swa_labels, swa_probs, swa_modalities, args.threshold_calibration,
        )
        swa_threshold, swa_threshold_metrics = find_best_threshold(swa_threshold_labels, swa_threshold_probs)
        swa_full_val_metrics = metrics_from_predictions(
            swa_labels, (np.asarray(swa_probs) >= swa_threshold).astype(np.int64), swa_probs,
        )
        swa_full_val_metrics["threshold"] = swa_threshold
        swa_full_val_metrics["threshold_source"] = swa_threshold_source
        swa_full_val_metrics["swa_n_epochs_averaged"] = len(swa_state_queue)

        swa_path = args.output_dir / f"arkplus_tabular_{args.preprocess_type}_fold{fold}_swa.pth"
        torch.save(
            {
                "model_state_dict": swa_full_state,
                "threshold": swa_threshold,
                "threshold_source": swa_threshold_source,
                "metrics": swa_full_val_metrics,
                "swa_n_epochs_averaged": len(swa_state_queue),
                "age_mean": age_mean,
                "age_std": age_std,
                "tab_hidden": args.tab_hidden,
                "balance_modality_label": args.balance_modality_label,
                "class_weight": args.class_weight,
                "label_smoothing": args.label_smoothing,
                "threshold_calibration": args.threshold_calibration,
                "checkpoint_metric": args.checkpoint_metric,
                "fine_tune_last_stage": args.fine_tune_last_stage,
            },
            swa_path,
        )
        print(
            f"Saved SWA model (avg of last {len(swa_state_queue)} epochs) to: {swa_path}  "
            f"Val F1@Thr: {swa_full_val_metrics['f1']:.4f} (best-checkpoint Val F1@Thr: {best_f1:.4f})"
        )

    return best_metrics, best_path


@torch.no_grad()
def load_checkpoint_model_and_threshold(path, args, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    backbone = build_arkplus_swin_base224(num_classes=NUM_CLASSES)
    backbone = load_arkplus_base224_weights(backbone, args.ark_checkpoint)
    for p in backbone.parameters():
        p.requires_grad = False
    model = TabularFusionModel(
        backbone, tabular_dim=2, num_classes=NUM_CLASSES,
        tab_hidden=checkpoint.get("tab_hidden", args.tab_hidden),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(checkpoint.get("threshold", 0.5))
    age_mean = checkpoint["age_mean"]
    age_std = checkpoint["age_std"]
    return model, threshold, age_mean, age_std


def resolve_image_ensemble_paths(args):
    if not args.image_ensemble_glob:
        return []
    paths = sorted(args.image_ensemble_dir.glob(args.image_ensemble_glob))
    if not paths:
        raise FileNotFoundError(
            f"No image ensemble weights matched: {args.image_ensemble_dir / args.image_ensemble_glob}"
        )
    return paths


def remap_legacy_swin_state_dict(state_dict, model, prefix=""):
    model_state = model.state_dict()
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        head_key = f"{prefix}head.fc."
        if new_key.startswith(head_key):
            new_key = new_key.replace(head_key, f"{prefix}head.", 1)

        layer_prefix = f"{prefix}layers."
        if ".downsample." in new_key and new_key.startswith(layer_prefix):
            parts = new_key.split(".")
            layer_part_index = 1 if prefix == "" else 2
            try:
                layer_idx = int(parts[layer_part_index])
                candidate_parts = parts.copy()
                candidate_parts[layer_part_index] = str(layer_idx - 1)
                candidate = ".".join(candidate_parts)
                if layer_idx > 0 and candidate in model_state and value.shape == model_state[candidate].shape:
                    new_key = candidate
            except ValueError:
                pass

        if new_key in model_state and value.shape == model_state[new_key].shape:
            remapped[new_key] = value
    return remapped


@torch.no_grad()
def load_compatible_image_checkpoint(path, args, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_arkplus_swin_base224(num_classes=NUM_CLASSES)
    model = load_arkplus_base224_weights(model, args.ark_checkpoint)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    state_dict = remap_legacy_swin_state_dict(state_dict, model)
    msg = model.load_state_dict(state_dict, strict=False)
    threshold = float(checkpoint.get("threshold", 0.5)) if isinstance(checkpoint, dict) else 0.5
    print(f"Loaded image checkpoint {path.name}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} threshold={threshold:.2f}")
    model.eval().to(device)
    return model, threshold


@torch.no_grad()
def collect_image_ensemble_probs(model_paths, test_df, transform, reference_quantiles, args, device):
    if not model_paths:
        return None, []

    print(f"Image ensemble models: {len(model_paths)}")
    image_ds = BiasMitigatedTBNormalDataset(
        test_df,
        args.test_img_dir,
        args.preprocess_type,
        transform,
        artifact_clean=args.artifact_clean,
        reference_quantiles=reference_quantiles,
        return_meta=True,
    )
    image_loader = DataLoader(
        image_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    image_models, image_thresholds = [], []
    for path in model_paths:
        model, threshold = load_compatible_image_checkpoint(path, args, device)
        image_models.append(model)
        image_thresholds.append(threshold)

    probs_all = []
    for images, _, _ in image_loader:
        images = images.to(device, non_blocking=True)
        logits = torch.stack([model(images) for model in image_models]).mean(dim=0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        probs_all.extend(probs.cpu().tolist())

    return np.asarray(probs_all, dtype=np.float32), image_thresholds


def resolve_eval_tabular_paths(args):
    if not args.eval_weights_glob:
        return None
    paths = sorted(args.eval_weights_dir.glob(args.eval_weights_glob))
    if not paths:
        raise FileNotFoundError(
            f"No tabular evaluation weights matched: {args.eval_weights_dir / args.eval_weights_glob}"
        )
    return paths


@torch.no_grad()
def evaluate_test_ensemble(model_paths, test_df, transform, reference_quantiles, args, device):
    print("\n========== Ark+ Tabular-Fusion Internal Test Evaluation ==========")
    print(f"Test CSV rows    : {len(test_df)}")
    print(f"Ensemble models  : {len(model_paths)}")

    models, thresholds = [], []
    age_mean = age_std = None
    for path in model_paths:
        model, threshold, age_mean, age_std = load_checkpoint_model_and_threshold(path, args, device)
        models.append(model)
        thresholds.append(threshold)

    test_ds = TabularFusionDataset(
        test_df, args.test_img_dir, args.preprocess_type, transform,
        age_mean, age_std, artifact_clean=args.artifact_clean,
        reference_quantiles=reference_quantiles, return_meta=True,
    )
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    loss_sum, n = 0.0, 0
    labels_all, probs_all, modalities_all = [], [], []
    for images, tabular, labels, modalities in test_loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)
        logits = torch.stack([model(images, tabular) for model in models]).mean(dim=0)
        loss = criterion(logits, labels_dev)
        probs = torch.softmax(logits, dim=1)[:, 1]
        loss_sum += loss.item() * images.size(0)
        n += images.size(0)
        labels_all.extend(labels.tolist())
        probs_all.extend(probs.cpu().tolist())
        modalities_all.extend(list(modalities))

    probs_all = np.asarray(probs_all, dtype=np.float32)
    image_probs, image_thresholds = collect_image_ensemble_probs(
        resolve_image_ensemble_paths(args),
        test_df,
        transform,
        reference_quantiles,
        args,
        device,
    )
    if image_probs is not None:
        weight = args.image_ensemble_weight
        probs_all = (1.0 - weight) * probs_all + weight * image_probs
        print(f"Blended image ensemble probability weight: {weight:.2f}")

    cv_threshold = float(np.mean(thresholds)) if thresholds else 0.5
    if image_probs is not None and args.image_ensemble_weight >= 1.0 and image_thresholds:
        cv_threshold = float(np.mean(image_thresholds))
    default_metrics = metrics_from_predictions(labels_all, (probs_all >= 0.5).astype(np.int64), probs_all)
    cv_metrics = metrics_from_predictions(labels_all, (probs_all >= cv_threshold).astype(np.int64), probs_all)
    internal_threshold, internal_best_metrics = find_best_threshold(labels_all, probs_all)

    print(f"Test Loss: {loss_sum / n if n else 0.0:.4f}")
    print_metric_line("Internal test default", default_metrics, 0.5)
    print_metric_line("Internal test calibrated", cv_metrics, cv_threshold)
    print_metric_line("Internal test best-threshold", internal_best_metrics, internal_threshold)
    print_per_modality_metrics("Internal test default", labels_all, probs_all, modalities_all, 0.5)
    print_per_modality_metrics("Internal test calibrated", labels_all, probs_all, modalities_all, cv_threshold)
    print_per_modality_metrics("Internal test best-threshold", labels_all, probs_all, modalities_all, internal_threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Ark+ tabular fusion (age+gender, no modality) -- R&D only.")
    parser.add_argument("--ark-checkpoint", default=Path("models/arkplus_pretrained/ark6_swinbase_224_ep200.pth.repacked.pth"), type=Path)
    parser.add_argument("--train-csv", default=Path("data/internal/Data/train.csv"), type=Path)
    parser.add_argument("--test-csv", default=Path("data/internal/Data/test.csv"), type=Path)
    parser.add_argument("--train-img-dir", default=Path("data/internal/Data/train/train"), type=Path)
    parser.add_argument("--test-img-dir", default=Path("data/internal/Data/test/test"), type=Path)
    parser.add_argument("--output-dir", default=Path("experiments/repeat_baseline_seed42/weights"), type=Path)
    parser.add_argument("--image-ensemble-dir", default=Path("experiments/arkplus_lastfinetune/weights"), type=Path)
    parser.add_argument("--image-ensemble-glob", default=None)
    parser.add_argument(
        "--image-ensemble-weight",
        default=0.0,
        type=float,
        help="Blend weight for optional image-only last-stage ensemble probabilities. Use 1.0 for image-only.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-weights-dir", default=Path("experiments/repeat_baseline_seed42/weights"), type=Path)
    parser.add_argument("--eval-weights-glob", default=None)
    parser.add_argument("--preprocess-type", default="ch0", choices=["ch0", "ch1", "ch2"])
    parser.add_argument("--train-modality-filter", default="all")
    parser.add_argument("--test-modality-filter", default="all")
    parser.add_argument("--artifact-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--histogram-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--histogram-reference-modality", default="CR")
    parser.add_argument("--histogram-reference-max-images", default=1200, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--k-folds", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--val-batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--tab-hidden", default=64, type=int)
    parser.add_argument(
        "--swa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save a Stochastic Weight Averaging checkpoint (_swa.pth) per fold: the "
             "trainable head's weights averaged over the last --swa-last-n-epochs epochs. "
             "Backbone is frozen so only the tabular MLP + classifier are averaged. Purely "
             "additive -- does not change _best.pth/_last.pth selection or saving.",
    )
    parser.add_argument("--swa-last-n-epochs", default=5, type=int,
                         help="Number of trailing epochs to average for the SWA checkpoint.")
    parser.add_argument(
        "--balance-modality-label",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance Modality_DICOM x TB/Normal groups during training. Enabled by default.",
    )
    parser.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Use inverse-frequency class weights in the training loss.",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="CrossEntropy label smoothing. Small smoothing usually stabilizes thresholded F1.",
    )
    parser.add_argument(
        "--threshold-calibration",
        default="all",
        choices=["all", "cr"],
        help="Validation subset used to pick the F1 threshold. Use 'all' for internal F1, 'cr' for CR-only calibration.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        default="full-val-f1",
        choices=["full-val-f1", "threshold-source-f1"],
        help="Metric used for best-checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--fine-tune-last-stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unfreeze Swin layers.3 and norm. Enabled by default to target >0.99 internal F1.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compatibility flag for loading optional image-only ensemble checkpoints.",
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    if not args.ark_checkpoint.exists():
        parser.error(f"Ark+ checkpoint not found: {args.ark_checkpoint}")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = build_transform()

    raw_df = pd.read_csv(args.train_csv)
    raw_test_df = pd.read_csv(args.test_csv)
    print_modality_table("Raw train", raw_df)
    print_modality_table("Raw test", raw_test_df)

    df = filter_by_modality(raw_df, args.train_modality_filter)
    test_df = filter_by_modality(raw_test_df, args.test_modality_filter)

    age_mean = float(df["age"].mean())
    age_std = float(df["age"].std())
    print(f"\nAge normalization: mean={age_mean:.2f} std={age_std:.2f} (from training data)")
    print("Tabular features: age (z-normalized), gender (binary). Modality_DICOM EXCLUDED by design.")

    reference_quantiles = build_reference_quantiles(df, args.train_img_dir, args)

    print("\n========== Ark+ Tabular-Fusion Dataset Counts ==========")
    print(f"Train CSV rows       : {len(df)}")
    print(f"Training PNG files   : {count_pngs(args.train_img_dir)}")
    print(f"Test CSV rows        : {len(test_df)}")
    print(f"Test PNG files       : {count_pngs(args.test_img_dir)}")
    print(f"Modality balancing   : {args.balance_modality_label}")
    print(f"Class loss weighting : {args.class_weight}")
    print(f"Label smoothing      : {args.label_smoothing}")
    print(f"Threshold calibration: {args.threshold_calibration}")
    print(f"Checkpoint metric    : {args.checkpoint_metric}")
    print(f"Fine-tune last stage : {args.fine_tune_last_stage}")
    print(f"Image ensemble glob  : {args.image_ensemble_glob}")
    print(f"Image ensemble weight: {args.image_ensemble_weight}")
    print(f"Device               : {device}")

    eval_paths = resolve_eval_tabular_paths(args)
    if args.eval_only:
        if eval_paths is None:
            raise ValueError("--eval-only requires --eval-weights-glob.")
        evaluate_test_ensemble(eval_paths, test_df, transform, reference_quantiles, args, device)
        return

    y = df["TB/Normal"].astype(str).str.strip().str.lower()
    splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    fold_metrics, best_paths = [], []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, y), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        metrics, best_path = run_fold(fold, train_df, val_df, transform, age_mean, age_std,
                                       reference_quantiles, args, device)
        fold_metrics.append(metrics)
        best_paths.append(best_path)
        gc.collect()
        torch.cuda.empty_cache()

    print_cv_summary(fold_metrics)
    evaluate_test_ensemble(best_paths, test_df, transform, reference_quantiles, args, device)


if __name__ == "__main__":
    main()
