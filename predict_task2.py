# Author: Sampa Misra, Glee lab

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import timm

try:
    from border_marker_cleanup import clean_gray_array
except ModuleNotFoundError:
    clean_gray_array = None


IMAGE_SIZE = 224
BATCH_SIZE = 16
ID2LABEL = {0: "Normal", 1: "TB"}

# Narrow-band TTA: only re-examine predictions this close to the decision
# boundary, with four extra zoom/brightness views, and average. Predictions
# outside this band are left untouched. Horizontal flip is deliberately NOT
# used -- chest X-ray anatomy is not left-right symmetric. Zoom and
# brightness were chosen over rotation after local validation showed they
# never regressed on any known external dataset (rotation showed small
# losses on 2/3 tested domains); brightness variation also has a direct
# mechanistic link to cross-scanner exposure differences.
TTA_LOW = 0.45
TTA_HIGH = 0.62
TTA_ZOOM_BRIGHTNESS_VIEWS = [
    (1.05, 1.0),  # zoom in 5%
    (0.95, 1.0),  # zoom out 5%
    (1.0, 1.1),   # brighten 10%
    (1.0, 0.9),   # darken 10%
]


def rescale_to_01(data, lower=0.5, upper=99.5):
    data = np.asarray(data).astype(np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.percentile(data, lower)
    hi = np.percentile(data, upper)
    if hi <= lo + 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    data = np.clip(data, lo, hi)
    return ((data - lo) / (hi - lo + 1e-8)).astype(np.float32)


def autocrop_letterbox_fallback(gray_u8, black_thresh=8, min_content_frac=0.6):
    h, w = gray_u8.shape
    row_max = gray_u8.max(axis=1)
    col_max = gray_u8.max(axis=0)
    rows = np.where(row_max > black_thresh)[0]
    cols = np.where(col_max > black_thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray_u8
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    if (bottom - top) < min_content_frac * h or (right - left) < min_content_frac * w:
        return gray_u8
    return gray_u8[top:bottom, left:right]


def clean_artifacts(gray_u8):
    if clean_gray_array is not None:
        return clean_gray_array(gray_u8, do_crop=True, do_mask=True)
    return autocrop_letterbox_fallback(gray_u8)


def match_to_reference_quantiles(gray_u8, reference_quantiles):
    if reference_quantiles is None:
        return gray_u8
    gray = np.asarray(gray_u8).astype(np.float32)
    q = np.linspace(0, 100, len(reference_quantiles))
    src = np.percentile(gray, q)
    unique_src, unique_idx = np.unique(src, return_index=True)
    if len(unique_src) < 2:
        return gray_u8
    target = np.asarray(reference_quantiles, dtype=np.float32)[unique_idx]
    matched = np.interp(gray.ravel(), unique_src, target).reshape(gray.shape)
    return np.clip(matched, 0, 255).astype(np.uint8)


def image_to_rgb(img_path, reference_quantiles):
    raw = Image.open(img_path).convert("L")
    gray = np.asarray(raw).astype(np.uint8)
    gray = clean_artifacts(gray)
    gray = match_to_reference_quantiles(gray, reference_quantiles)
    gray = gray.astype(np.float32)
    base_norm = (gray - np.mean(gray)) / (np.std(gray) + 1e-8)
    processed = rescale_to_01(base_norm)
    processed = (np.clip(processed, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(processed).convert("RGB")


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class TabularFusionArkPlus(nn.Module):
    def __init__(self, tab_hidden=16, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
        )
        if hasattr(self.backbone.head, "fc"):
            img_feat_dim = self.backbone.head.fc.in_features
        else:
            img_feat_dim = self.backbone.head.in_features
        self.tabular_mlp = nn.Sequential(
            nn.Linear(2, tab_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(img_feat_dim + tab_hidden, num_classes)

    def forward(self, image, tabular):
        feats = self.backbone.forward_features(image)
        pooled = self.backbone.forward_head(feats, pre_logits=True)
        tab = self.tabular_mlp(tabular)
        return self.classifier(torch.cat([pooled, tab], dim=1))


def remap_state_dict(state_dict, model):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    remapped = {}
    for key, value in state_dict.items():
        if "attn_mask" in key or "relative_position_index" in key:
            continue
        new_key = key
        if key in model_state and value.shape == model_state[key].shape:
            pass
        elif key == "head.fc.weight":
            new_key = "head.weight"
        elif key == "head.fc.bias":
            new_key = "head.bias"
        elif ".downsample." in key and key.startswith("layers."):
            parts = key.split(".")
            try:
                layer_idx = int(parts[1])
                candidate = ".".join([parts[0], str(layer_idx - 1), *parts[2:]])
                if layer_idx > 0 and candidate in model_state and value.shape == model_state[candidate].shape:
                    new_key = candidate
            except ValueError:
                pass
        if new_key in model_state and value.shape == model_state[new_key].shape:
            remapped[new_key] = value
    return remapped


def load_tabular_model(path, device, tab_hidden):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = TabularFusionArkPlus(tab_hidden=checkpoint.get("tab_hidden", tab_hidden))
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    msg = model.load_state_dict(remap_state_dict(state, model), strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"{Path(path).name}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval().to(device)
    age_mean = float(checkpoint.get("age_mean", 49.05))
    age_std = float(checkpoint.get("age_std", 15.55))
    return model, age_mean, age_std


def load_tabular_ensemble(weights_dir, weights_glob, device, tab_hidden):
    paths = sorted(glob.glob(os.path.join(weights_dir, weights_glob)))
    if not paths:
        raise RuntimeError(f"No tabular-fusion checkpoints found: {weights_dir}/{weights_glob}")
    models = [load_tabular_model(path, device, tab_hidden) for path in paths]
    print(f"Loaded tabular-fusion Ark+ ensemble: {len(models)} folds")
    return models


def parse_age(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "na", "unknown"}:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    age = float(match.group(0))
    if "month" in text or text.endswith("mo"):
        age /= 12.0
    return age


def parse_gender(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"male", "m", "man"}:
        return 1.0
    if text in {"female", "f", "woman"}:
        return 0.0
    return None


def row_key(row):
    for col in ("filename", "file", "image", "image_name", "path"):
        if col in row and row[col]:
            return Path(str(row[col])).name
    for col in ("new_id", "id", "study_id"):
        if col in row and row[col]:
            value = str(row[col]).strip()
            return value if value.lower().endswith(".png") else f"{value}.png"
    return None


def load_metadata(input_dir):
    metadata = {}
    for csv_path in sorted(Path(input_dir).glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row_key(row)
                if not key:
                    continue
                age = None
                for col in ("age", "Age", "patient_age", "PatientAge"):
                    if col in row:
                        age = parse_age(row[col])
                        break
                gender = None
                for col in ("gender", "Gender", "sex", "Sex", "patient_sex", "PatientSex"):
                    if col in row:
                        gender = parse_gender(row[col])
                        break
                meta = {"age": age, "gender": gender}
                metadata[key] = meta
                metadata[Path(key).stem] = meta
    print(f"Metadata rows matched by filename/id: {len(metadata)}")
    return metadata


def tabular_for_model(meta, age_mean, age_std, device):
    if meta is None:
        age_norm = 0.0
        gender = 0.5
    else:
        age = meta.get("age")
        gender = meta.get("gender")
        age_norm = 0.0 if age is None else (float(age) - age_mean) / (age_std + 1e-8)
        gender = 0.5 if gender is None else float(gender)
    return torch.tensor([age_norm, gender], dtype=torch.float32, device=device)


@torch.no_grad()
def ensemble_prob(tabular_models, images, batch_paths, metadata, device):
    # Harmonic mean of each fold's own softmax probability. A power-mean sweep
    # (p from -3 to +3, where geometric mean is the p=0 limit and harmonic mean
    # is p=-1) found harmonic mean sits exactly at the edge of the "free lunch"
    # zone: pushing further toward disagreement-penalization than geomean keeps
    # helping the hardest external domains up through p=-1, but going past that
    # point starts costing other domains real ground. Verified with the exact
    # production resize/TTA pipeline: zero regressions vs geomean on any of 6
    # locally tested datasets, real gains on the two hardest (TBX11K, TB Chest
    # Radiography).
    fold_probs = []
    for model, age_mean, age_std in tabular_models:
        tabular = []
        for path in batch_paths:
            name = Path(path).name
            meta = metadata.get(name) or metadata.get(Path(name).stem)
            tabular.append(tabular_for_model(meta, age_mean, age_std, device))
        tabular = torch.stack(tabular)
        logits = model(images, tabular)
        fold_probs.append(torch.softmax(logits, dim=1)[:, 1])
    fold_probs = torch.stack(fold_probs)
    return 1.0 / (1.0 / fold_probs.clamp_min(1e-8)).mean(dim=0)


def make_zoom_brightness_view(rgb, zoom, brightness):
    img = rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
    if brightness != 1.0:
        img = TF.adjust_brightness(img, brightness)
    if zoom != 1.0:
        new_size = int(round(IMAGE_SIZE * zoom))
        img = img.resize((new_size, new_size))
        if zoom > 1.0:
            left = (new_size - IMAGE_SIZE) // 2
            img = img.crop((left, left, left + IMAGE_SIZE, left + IMAGE_SIZE))
        else:
            pad = (IMAGE_SIZE - new_size) // 2
            canvas = Image.new(img.mode, (IMAGE_SIZE, IMAGE_SIZE), 0)
            canvas.paste(img, (pad, pad))
            img = canvas
    t = TF.to_tensor(img)
    return TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


@torch.no_grad()
def predict_all(tabular_models, transform, reference_quantiles, image_paths, metadata, threshold, device):
    base_probs = {}
    cached_rgb = {}
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Inference"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        rgb_images = [image_to_rgb(path, reference_quantiles) for path in batch_paths]
        images = torch.stack([transform(rgb) for rgb in rgb_images]).to(device)

        tb_probs = ensemble_prob(tabular_models, images, batch_paths, metadata, device)
        for path, rgb, tb_prob in zip(batch_paths, rgb_images, tb_probs.cpu().numpy()):
            base_probs[path] = float(tb_prob)
            if TTA_LOW <= tb_prob <= TTA_HIGH:
                cached_rgb[path] = rgb

    final_probs = dict(base_probs)
    borderline_paths = list(cached_rgb.keys())
    if borderline_paths:
        print(f"Narrow-band TTA: re-examining {len(borderline_paths)} borderline prediction(s) "
              f"(p in [{TTA_LOW}, {TTA_HIGH}]) with {len(TTA_ZOOM_BRIGHTNESS_VIEWS)} zoom/brightness views.")
        for i in tqdm(range(0, len(borderline_paths), BATCH_SIZE), desc="TTA"):
            batch_paths = borderline_paths[i : i + BATCH_SIZE]
            rgb_images = [cached_rgb[path] for path in batch_paths]

            view_probs = [torch.as_tensor([base_probs[path] for path in batch_paths])]
            for zoom, brightness in TTA_ZOOM_BRIGHTNESS_VIEWS:
                views = torch.stack([make_zoom_brightness_view(rgb, zoom, brightness) for rgb in rgb_images]).to(device)
                view_probs.append(ensemble_prob(tabular_models, views, batch_paths, metadata, device).cpu())

            tta_prob = torch.stack(view_probs).mean(dim=0)
            for path, prob in zip(batch_paths, tta_prob.tolist()):
                final_probs[path] = prob

    rows = []
    for path in image_paths:
        pred = 1 if final_probs[path] >= threshold else 0
        rows.append([Path(path).name, ID2LABEL[pred]])
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 final class-weighted metadata-aware Ark+ fusion inference")
    parser.add_argument("--input", default="/input", help="folder with test PNGs and optional metadata CSV")
    parser.add_argument("--output", default="/output", help="folder where prediction.csv is written")
    parser.add_argument("--weights-dir", default="/workspace/weights")
    parser.add_argument("--tabular-weights-glob", default="class_weighted_metadata_fusion/arkplus_tabular_ch0_fold*_best.pth")
    parser.add_argument("--reference-quantiles", default="/workspace/weights/reference_quantiles_ch0.npy")
    parser.add_argument("--tab-hidden", default=64, type=int)
    parser.add_argument("--threshold", default=0.50, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
    if not image_paths:
        raise RuntimeError(f"No PNG images found under: {args.input}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reference_quantiles = np.load(args.reference_quantiles)
    metadata = load_metadata(args.input)
    tabular_models = load_tabular_ensemble(args.weights_dir, args.tabular_weights_glob, device, args.tab_hidden)
    transform = build_transform()
    rows = predict_all(tabular_models, transform, reference_quantiles, image_paths, metadata, args.threshold, device)

    for csv_name in ("prediction.csv", "test.csv"):
        csv_path = os.path.join(args.output, csv_name)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "TB/Normal"])
            writer.writerows(rows)
        print(f"Wrote {len(rows)} predictions to {csv_path}")


if __name__ == "__main__":
    main()
