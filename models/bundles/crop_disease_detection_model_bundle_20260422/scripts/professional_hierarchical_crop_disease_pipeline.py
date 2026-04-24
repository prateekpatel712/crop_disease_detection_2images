#!/usr/bin/env python3
"""Business-grade crop disease pipeline on top of the saved joint model.

Goals:
- keep all 134 crop-disease classes
- avoid optimistic evaluation caused by duplicate-style stem leakage
- clean only conservatively: relabel clear majority-conflict groups and
  downweight suspicious samples instead of deleting classes
- train crop-specific disease heads on penultimate features
- tune the final joint + hierarchical blend on validation with macro-F1 as
  the primary objective
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


DATASET_ROOT = Path("/home/jupyter/combined_plant_dataset_normalized")
PIPELINE_ROOT = DATASET_ROOT / "professional_all_crop_disease_outputs"
PREP_ROOT = PIPELINE_ROOT / "prepared_data"
TRAIN_CSV = PREP_ROOT / "train_manifest.csv"
VAL_CSV = PREP_ROOT / "val_manifest.csv"
TEST_CSV = PREP_ROOT / "test_manifest.csv"
ID_TO_LABEL_JSON = PREP_ROOT / "id_to_label.json"
CHECKPOINT = PIPELINE_ROOT / "training_outputs" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "professional_hierarchical_outputs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a professional hierarchical crop disease pipeline.")
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=VAL_CSV)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--id-to-label-json", type=Path, default=ID_TO_LABEL_JSON)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--io-threads", type=int, default=8)
    parser.add_argument("--max-train-per-crop", type=int, default=None)
    parser.add_argument("--max-val-per-crop", type=int, default=None)
    parser.add_argument("--max-test-per-crop", type=int, default=None)
    parser.add_argument("--majority-relabel-threshold", type=float, default=0.60)
    parser.add_argument("--cleanup-margin-center", type=float, default=0.02)
    parser.add_argument("--cleanup-margin-scale", type=float, default=0.08)
    parser.add_argument("--min-cleanup-weight", type=float, default=0.15)
    parser.add_argument("--c-grid", type=str, default="0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--crop-scale-grid", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--blend-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--joint-alpha-grid", type=str, default="-0.5,0.0,0.5,1.0,1.5")
    parser.add_argument("--joint-beta-grid", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--max-iter", type=int, default=3000)
    return parser.parse_args()


def parse_float_grid(spec: str) -> List[float]:
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def canonical_stem(path_value: str) -> str:
    stem = Path(path_value).stem.lower()
    return re.sub(r"(?:__dup\d+)+$", "", stem)


def add_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["canonical_stem"] = df["image_path"].map(canonical_stem)
    df["stem_key"] = list(zip(df["crop_label"], df["canonical_stem"]))
    return df


def maybe_cap_per_crop(df: pd.DataFrame, max_per_crop: int | None) -> pd.DataFrame:
    if max_per_crop is None:
        return df.reset_index(drop=True)
    return df.groupby("crop_label", group_keys=False).head(max_per_crop).reset_index(drop=True)


def load_id_to_label(path: Path) -> Dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def build_label_maps(id_to_label: Dict[int, str]) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    label_to_id = {label: label_id for label_id, label in id_to_label.items()}
    crop_to_ids: Dict[str, List[int]] = defaultdict(list)
    for label_id, label in id_to_label.items():
        crop = label.split("::", 1)[0]
        crop_to_ids[crop].append(label_id)
    for ids in crop_to_ids.values():
        ids.sort()
    return label_to_id, dict(crop_to_ids)


@dataclass
class CleanupReport:
    original_train_rows: int
    removed_leakage_rows: int
    relabeled_conflict_rows: int
    unresolved_conflict_rows: int
    train_conflict_groups: int
    cross_split_leakage_groups: int


def cleanup_manifests(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_to_id: Dict[str, int],
    majority_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CleanupReport]:
    train_df = add_canonical_columns(train_df)
    val_df = add_canonical_columns(val_df)
    test_df = add_canonical_columns(test_df)

    heldout_keys = set(val_df["stem_key"].tolist()) | set(test_df["stem_key"].tolist())
    original_train_keys = set(train_df["stem_key"].tolist())
    cross_split_leakage_groups = len(original_train_keys & heldout_keys)
    leakage_mask = train_df["stem_key"].isin(heldout_keys)
    removed_leakage_rows = int(leakage_mask.sum())
    train_df = train_df.loc[~leakage_mask].reset_index(drop=True)

    train_df["cleaned_crop_disease_label"] = train_df["crop_disease_label"]
    train_df["cleaned_crop_disease_label_id"] = train_df["crop_disease_label_id"].astype(int)
    train_df["stem_conflict_unresolved"] = False
    train_df["stem_conflict_relabelled"] = False

    relabeled_conflict_rows = 0
    unresolved_conflict_rows = 0
    conflict_groups = 0

    for (_, _), group in train_df.groupby(["crop_label", "canonical_stem"], sort=False):
        counts = group["crop_disease_label"].value_counts()
        if len(counts) <= 1:
            continue
        conflict_groups += 1
        top_label = str(counts.index[0])
        top_count = int(counts.iloc[0])
        ratio = top_count / len(group)
        idx = group.index

        if ratio >= majority_threshold:
            minority_mask = train_df.loc[idx, "cleaned_crop_disease_label"] != top_label
            relabeled_conflict_rows += int(minority_mask.sum())
            train_df.loc[idx, "cleaned_crop_disease_label"] = top_label
            train_df.loc[idx, "cleaned_crop_disease_label_id"] = int(label_to_id[top_label])
            train_df.loc[idx, "stem_conflict_relabelled"] = True
        else:
            unresolved_conflict_rows += int(len(group))
            train_df.loc[idx, "stem_conflict_unresolved"] = True

    report = CleanupReport(
        original_train_rows=int(len(train_df) + removed_leakage_rows),
        removed_leakage_rows=removed_leakage_rows,
        relabeled_conflict_rows=relabeled_conflict_rows,
        unresolved_conflict_rows=unresolved_conflict_rows,
        train_conflict_groups=conflict_groups,
        cross_split_leakage_groups=cross_split_leakage_groups,
    )
    return train_df, val_df, test_df, report


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        with Image.open(row["image_path"]) as img:
            image = self.transform(img.convert("RGB"))
        return image, int(row["cleaned_crop_disease_label_id"] if "cleaned_crop_disease_label_id" in row else row["crop_disease_label_id"])


class EfficientNetFeatureModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.30, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.model.features(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.classifier(features)
        return features, logits


def build_feature_model(checkpoint_path: Path, num_classes: int) -> EfficientNetFeatureModel:
    model = EfficientNetFeatureModel(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_preprocessed_image(path_value: str, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(path_value) as img:
        return transform(img.convert("RGB"))


@dataclass
class FeatureBundle:
    features: np.ndarray
    logits: np.ndarray
    targets: np.ndarray


def extract_feature_bundle(
    model: EfficientNetFeatureModel,
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    io_threads: int,
    cache_path: Path,
) -> FeatureBundle:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return FeatureBundle(
            features=payload["features"].numpy(),
            logits=payload["logits"].numpy(),
            targets=payload["targets"].numpy(),
        )

    features_parts: List[torch.Tensor] = []
    logits_parts: List[torch.Tensor] = []
    target_parts: List[torch.Tensor] = []
    transform = build_eval_transform(image_size)
    paths = df["image_path"].tolist()
    targets = (
        df["cleaned_crop_disease_label_id"].astype(int).tolist()
        if "cleaned_crop_disease_label_id" in df.columns
        else df["crop_disease_label_id"].astype(int).tolist()
    )

    with ThreadPoolExecutor(max_workers=io_threads) as executor, torch.no_grad():
        progress = tqdm(range(0, len(df), batch_size), desc=f"extract {cache_path.stem}", dynamic_ncols=True)
        for start in progress:
            end = min(start + batch_size, len(df))
            batch_paths = paths[start:end]
            batch_targets = torch.as_tensor(targets[start:end], dtype=torch.long)
            images = list(executor.map(lambda p: load_preprocessed_image(p, transform), batch_paths))
            images = torch.stack(images, dim=0)
            features, logits = model(images)
            features_parts.append(features.cpu())
            logits_parts.append(logits.cpu())
            target_parts.append(batch_targets)

    features_tensor = torch.cat(features_parts, dim=0)
    logits_tensor = torch.cat(logits_parts, dim=0)
    targets_tensor = torch.cat(target_parts, dim=0)
    torch.save(
        {
            "features": features_tensor,
            "logits": logits_tensor,
            "targets": targets_tensor,
        },
        cache_path,
    )
    return FeatureBundle(
        features=features_tensor.numpy(),
        logits=logits_tensor.numpy(),
        targets=targets_tensor.numpy(),
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class CleanupWeightReport:
    suspicious_rows: int
    severe_suspicious_rows: int
    min_weight: float
    mean_weight: float


def build_cleanup_weights(
    train_df: pd.DataFrame,
    train_features: np.ndarray,
    crop_to_ids: Dict[str, List[int]],
    margin_center: float,
    margin_scale: float,
    min_weight: float,
) -> Tuple[np.ndarray, CleanupWeightReport]:
    x = l2_normalize(train_features)
    weights = np.ones(len(train_df), dtype=np.float64)
    severe = np.zeros(len(train_df), dtype=bool)
    suspicious = np.zeros(len(train_df), dtype=bool)

    train_labels = train_df["cleaned_crop_disease_label_id"].astype(int).to_numpy()
    train_crops = train_df["crop_label"].to_numpy()

    for crop, class_ids in crop_to_ids.items():
        mask = train_crops == crop
        if not np.any(mask):
            continue

        crop_x = x[mask]
        crop_y = train_labels[mask]
        class_to_local = {class_id: idx for idx, class_id in enumerate(class_ids)}

        prototypes = []
        for class_id in class_ids:
            class_mask = crop_y == class_id
            if np.any(class_mask):
                proto = crop_x[class_mask].mean(axis=0)
            else:
                proto = np.zeros(crop_x.shape[1], dtype=np.float64)
            norm = np.linalg.norm(proto)
            if norm > 0:
                proto = proto / norm
            prototypes.append(proto)
        prototypes = np.asarray(prototypes, dtype=np.float64)

        sims = crop_x @ prototypes.T
        local_true = np.asarray([class_to_local[int(label)] for label in crop_y], dtype=np.int64)
        true_sim = sims[np.arange(len(crop_y)), local_true]

        sims_without_true = sims.copy()
        sims_without_true[np.arange(len(crop_y)), local_true] = -1e9
        best_other = sims_without_true.max(axis=1)
        pred_local = sims.argmax(axis=1)
        pred_global = np.asarray([class_ids[idx] for idx in pred_local], dtype=np.int64)
        margin = true_sim - best_other

        local_weights = min_weight + (1.0 - min_weight) * sigmoid((margin - margin_center) / margin_scale)
        local_weights = np.clip(local_weights, min_weight, 1.0)

        unresolved = train_df.loc[mask, "stem_conflict_unresolved"].to_numpy(dtype=bool)
        relabelled = train_df.loc[mask, "stem_conflict_relabelled"].to_numpy(dtype=bool)
        local_weights[unresolved] *= 0.60
        local_weights[relabelled] *= 0.85
        local_weights = np.clip(local_weights, min_weight, 1.0)

        suspicious_mask = pred_global != crop_y
        severe_mask = suspicious_mask & ((best_other - true_sim) > 0.05)

        global_idx = np.flatnonzero(mask)
        weights[global_idx] = local_weights
        suspicious[global_idx] = suspicious_mask
        severe[global_idx] = severe_mask

    report = CleanupWeightReport(
        suspicious_rows=int(suspicious.sum()),
        severe_suspicious_rows=int(severe.sum()),
        min_weight=float(weights.min()),
        mean_weight=float(weights.mean()),
    )
    return weights, report


def weighted_selection_score(macro_f1: float, accuracy: float) -> float:
    return 0.70 * macro_f1 + 0.30 * accuracy


@dataclass
class CropModelSelection:
    crop: str
    selected_c: float
    val_accuracy: float
    val_macro_f1: float
    class_ids: List[int]
    model_path: str


def fit_crop_specific_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_weights: np.ndarray,
    crop_to_ids: Dict[str, List[int]],
    c_grid: Iterable[float],
    max_iter: int,
    output_dir: Path,
) -> List[CropModelSelection]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selections: List[CropModelSelection] = []

    train_y_all = train_df["cleaned_crop_disease_label_id"].astype(int).to_numpy()
    val_y_all = val_df["crop_disease_label_id"].astype(int).to_numpy()

    for crop, class_ids in sorted(crop_to_ids.items()):
        train_mask = train_df["crop_label"].to_numpy() == crop
        val_mask = val_df["crop_label"].to_numpy() == crop
        if not np.any(train_mask) or not np.any(val_mask):
            continue

        x_train = train_features[train_mask]
        x_val = val_features[val_mask]
        y_train_global = train_y_all[train_mask]
        y_val_global = val_y_all[val_mask]
        sample_weight = train_weights[train_mask]

        class_to_local = {class_id: idx for idx, class_id in enumerate(class_ids)}
        y_train = np.asarray([class_to_local[int(label)] for label in y_train_global], dtype=np.int64)
        y_val = np.asarray([class_to_local[int(label)] for label in y_val_global], dtype=np.int64)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        unique_train = np.unique(y_train)

        if len(unique_train) < 2:
            constant_local = int(unique_train[0])
            val_pred = np.full(len(y_val), constant_local, dtype=np.int64)
            val_acc = float(np.mean(val_pred == y_val))
            val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
            model_path = output_dir / f"{crop}_disease_head.joblib"
            joblib.dump(
                {
                    "kind": "constant",
                    "class_ids": class_ids,
                    "constant_local_class": constant_local,
                },
                model_path,
            )
            selections.append(
                CropModelSelection(
                    crop=crop,
                    selected_c=0.0,
                    val_accuracy=val_acc,
                    val_macro_f1=val_macro_f1,
                    class_ids=class_ids,
                    model_path=str(model_path),
                )
            )
            continue

        best = None
        best_payload = None
        for c_value in c_grid:
            clf = LogisticRegression(
                C=float(c_value),
                max_iter=max_iter,
                solver="lbfgs",
                class_weight="balanced",
            )
            clf.fit(x_train_scaled, y_train, sample_weight=sample_weight)
            val_pred = clf.predict(x_val_scaled)
            val_acc = float(np.mean(val_pred == y_val))
            val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
            score = weighted_selection_score(val_macro_f1, val_acc)
            candidate = {
                "c": float(c_value),
                "score": float(score),
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
                best_payload = {
                    "kind": "logreg",
                    "scaler": scaler,
                    "classifier": clf,
                    "class_ids": class_ids,
                    "seen_local_classes": [int(x) for x in clf.classes_.tolist()],
                }

        assert best is not None and best_payload is not None
        model_path = output_dir / f"{crop}_disease_head.joblib"
        joblib.dump(best_payload, model_path)
        selections.append(
            CropModelSelection(
                crop=crop,
                selected_c=float(best["c"]),
                val_accuracy=float(best["val_accuracy"]),
                val_macro_f1=float(best["val_macro_f1"]),
                class_ids=class_ids,
                model_path=str(model_path),
            )
        )
    return selections


def log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits_tensor = torch.from_numpy(logits)
    return torch.log_softmax(logits_tensor, dim=1).numpy()


def build_log_priors(train_df: pd.DataFrame, num_classes: int, crop_to_ids: Dict[str, List[int]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    counts = train_df["cleaned_crop_disease_label_id"].value_counts().sort_index()
    global_log_priors = np.log(
        np.asarray([(counts.get(i, 0) + 1e-9) / len(train_df) for i in range(num_classes)], dtype=np.float64)
    )

    crop_log_priors: Dict[str, np.ndarray] = {}
    for crop, class_ids in crop_to_ids.items():
        subcounts = train_df.loc[train_df["crop_label"] == crop, "cleaned_crop_disease_label_id"].value_counts()
        total = float(subcounts.sum())
        vec = np.full((num_classes,), -1e9, dtype=np.float64)
        for class_id in class_ids:
            vec[class_id] = math.log((subcounts.get(class_id, 0) + 1e-9) / total)
        crop_log_priors[crop] = vec
    return global_log_priors, crop_log_priors


def predict_crop_from_joint_scores(scores: np.ndarray, crop_to_ids: Dict[str, List[int]]) -> Tuple[List[str], np.ndarray]:
    crop_names = sorted(crop_to_ids)
    crop_scores = np.stack(
        [np.logaddexp.reduce(scores[:, crop_to_ids[crop]], axis=1) for crop in crop_names],
        axis=1,
    )
    pred_idx = crop_scores.argmax(axis=1)
    pred_crops = [crop_names[idx] for idx in pred_idx.tolist()]
    crop_log_probs = log_softmax_numpy(crop_scores)
    return pred_crops, crop_log_probs


def tune_joint_prior_adjustment(
    val_logits: np.ndarray,
    val_targets: np.ndarray,
    train_df: pd.DataFrame,
    crop_to_ids: Dict[str, List[int]],
    alpha_grid: Iterable[float],
    beta_grid: Iterable[float],
    num_classes: int,
) -> dict:
    global_priors, crop_priors = build_log_priors(train_df, num_classes=num_classes, crop_to_ids=crop_to_ids)
    best = None
    for alpha in alpha_grid:
        adjusted = val_logits + alpha * global_priors
        pred_crops, _ = predict_crop_from_joint_scores(adjusted, crop_to_ids)
        for beta in beta_grid:
            local_adjusted = adjusted.copy()
            for i, crop in enumerate(pred_crops):
                local_adjusted[i] = local_adjusted[i] + beta * crop_priors[crop]
            preds = local_adjusted.argmax(axis=1)
            acc = float(np.mean(preds == val_targets))
            macro_f1 = float(f1_score(val_targets, preds, average="macro"))
            score = weighted_selection_score(macro_f1, acc)
            candidate = {
                "alpha": float(alpha),
                "beta": float(beta),
                "val_accuracy": acc,
                "val_macro_f1": macro_f1,
                "selection_score": score,
            }
            if best is None or candidate["selection_score"] > best["selection_score"]:
                best = candidate
    assert best is not None
    return best


def apply_joint_prior_adjustment(
    logits: np.ndarray,
    train_df: pd.DataFrame,
    crop_to_ids: Dict[str, List[int]],
    alpha: float,
    beta: float,
    num_classes: int,
) -> np.ndarray:
    global_priors, crop_priors = build_log_priors(train_df, num_classes=num_classes, crop_to_ids=crop_to_ids)
    adjusted = logits + alpha * global_priors
    pred_crops, _ = predict_crop_from_joint_scores(adjusted, crop_to_ids)
    for i, crop in enumerate(pred_crops):
        adjusted[i] = adjusted[i] + beta * crop_priors[crop]
    return adjusted


def build_hierarchical_joint_scores(
    features: np.ndarray,
    crop_log_probs: np.ndarray,
    crop_to_ids: Dict[str, List[int]],
    crop_model_payloads: Dict[str, dict],
    crop_scale: float,
) -> np.ndarray:
    crop_names = sorted(crop_to_ids)
    crop_name_to_col = {crop: idx for idx, crop in enumerate(crop_names)}
    scores = np.full((features.shape[0], sum(len(v) for v in crop_to_ids.values())), -1e9, dtype=np.float64)

    for crop, class_ids in crop_to_ids.items():
        payload = crop_model_payloads[crop]
        if payload.get("kind") == "constant":
            disease_log_probs = np.full((features.shape[0], len(class_ids)), -1e9, dtype=np.float64)
            disease_log_probs[:, int(payload["constant_local_class"])] = 0.0
        else:
            scaler: StandardScaler = payload["scaler"]
            clf: LogisticRegression = payload["classifier"]
            x_scaled = scaler.transform(features)
            raw_log_probs = clf.predict_log_proba(x_scaled)
            disease_log_probs = np.full((features.shape[0], len(class_ids)), -1e9, dtype=np.float64)
            seen_local_classes = payload.get("seen_local_classes", list(range(raw_log_probs.shape[1])))
            for col_idx, local_class_idx in enumerate(seen_local_classes):
                disease_log_probs[:, int(local_class_idx)] = raw_log_probs[:, col_idx]
        crop_col = crop_name_to_col[crop]
        for local_idx, class_id in enumerate(class_ids):
            scores[:, class_id] = crop_scale * crop_log_probs[:, crop_col] + disease_log_probs[:, local_idx]
    return scores


@dataclass
class BlendSelection:
    crop_scale: float
    blend_weight: float
    val_accuracy: float
    val_macro_f1: float


def tune_hierarchical_blend(
    val_features: np.ndarray,
    val_targets: np.ndarray,
    adjusted_joint_logits: np.ndarray,
    crop_to_ids: Dict[str, List[int]],
    crop_model_payloads: Dict[str, dict],
    crop_scale_grid: Iterable[float],
    blend_grid: Iterable[float],
) -> BlendSelection:
    _, crop_log_probs = predict_crop_from_joint_scores(adjusted_joint_logits, crop_to_ids)
    adjusted_joint_log_probs = log_softmax_numpy(adjusted_joint_logits)

    best = None
    for crop_scale in crop_scale_grid:
        hier_scores = build_hierarchical_joint_scores(
            features=val_features,
            crop_log_probs=crop_log_probs,
            crop_to_ids=crop_to_ids,
            crop_model_payloads=crop_model_payloads,
            crop_scale=float(crop_scale),
        )
        hier_log_probs = log_softmax_numpy(hier_scores)
        for blend_weight in blend_grid:
            final_scores = blend_weight * adjusted_joint_log_probs + (1.0 - blend_weight) * hier_log_probs
            preds = final_scores.argmax(axis=1)
            acc = float(np.mean(preds == val_targets))
            macro_f1 = float(f1_score(val_targets, preds, average="macro"))
            score = weighted_selection_score(macro_f1, acc)
            candidate = BlendSelection(
                crop_scale=float(crop_scale),
                blend_weight=float(blend_weight),
                val_accuracy=acc,
                val_macro_f1=macro_f1,
            )
            if best is None or score > weighted_selection_score(best.val_macro_f1, best.val_accuracy):
                best = candidate

    assert best is not None
    return best


def evaluate_scores(scores: np.ndarray, targets: np.ndarray, id_to_label: Dict[int, str]) -> dict:
    preds = scores.argmax(axis=1)
    accuracy = float(np.mean(preds == targets))
    macro_f1 = float(f1_score(targets, preds, average="macro"))
    micro_f1 = float(f1_score(targets, preds, average="micro"))
    weighted_f1 = float(f1_score(targets, preds, average="weighted"))
    ordered_ids = sorted(id_to_label)
    ordered_labels = [id_to_label[i] for i in ordered_ids]
    report = classification_report(
        targets,
        preds,
        labels=ordered_ids,
        target_names=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(targets, preds, labels=ordered_ids)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "feature_cache"
    model_dir = args.output_dir / "per_crop_models"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    id_to_label = load_id_to_label(args.id_to_label_json)
    label_to_id, crop_to_ids = build_label_maps(id_to_label)

    train_df = maybe_cap_per_crop(pd.read_csv(args.train_csv, low_memory=False), args.max_train_per_crop)
    val_df = maybe_cap_per_crop(pd.read_csv(args.val_csv, low_memory=False), args.max_val_per_crop)
    test_df = maybe_cap_per_crop(pd.read_csv(args.test_csv, low_memory=False), args.max_test_per_crop)

    train_df, val_df, test_df, cleanup_report = cleanup_manifests(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_to_id=label_to_id,
        majority_threshold=args.majority_relabel_threshold,
    )

    model = build_feature_model(checkpoint_path=args.checkpoint, num_classes=len(id_to_label))

    train_bundle = extract_feature_bundle(
        model=model,
        df=train_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        io_threads=args.io_threads,
        cache_path=feature_dir / "train_features.pt",
    )
    val_bundle = extract_feature_bundle(
        model=model,
        df=val_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        io_threads=args.io_threads,
        cache_path=feature_dir / "val_features.pt",
    )
    test_bundle = extract_feature_bundle(
        model=model,
        df=test_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        io_threads=args.io_threads,
        cache_path=feature_dir / "test_features.pt",
    )

    cleanup_weights, cleanup_weight_report = build_cleanup_weights(
        train_df=train_df,
        train_features=train_bundle.features,
        crop_to_ids=crop_to_ids,
        margin_center=args.cleanup_margin_center,
        margin_scale=args.cleanup_margin_scale,
        min_weight=args.min_cleanup_weight,
    )

    crop_models = fit_crop_specific_models(
        train_df=train_df,
        val_df=val_df,
        train_features=train_bundle.features,
        val_features=val_bundle.features,
        train_weights=cleanup_weights,
        crop_to_ids=crop_to_ids,
        c_grid=parse_float_grid(args.c_grid),
        max_iter=args.max_iter,
        output_dir=model_dir,
    )
    crop_model_payloads = {item.crop: joblib.load(item.model_path) for item in crop_models}

    joint_selection = tune_joint_prior_adjustment(
        val_logits=val_bundle.logits,
        val_targets=val_bundle.targets,
        train_df=train_df,
        crop_to_ids=crop_to_ids,
        alpha_grid=parse_float_grid(args.joint_alpha_grid),
        beta_grid=parse_float_grid(args.joint_beta_grid),
        num_classes=len(id_to_label),
    )

    val_joint_adjusted = apply_joint_prior_adjustment(
        logits=val_bundle.logits,
        train_df=train_df,
        crop_to_ids=crop_to_ids,
        alpha=joint_selection["alpha"],
        beta=joint_selection["beta"],
        num_classes=len(id_to_label),
    )
    test_joint_adjusted = apply_joint_prior_adjustment(
        logits=test_bundle.logits,
        train_df=train_df,
        crop_to_ids=crop_to_ids,
        alpha=joint_selection["alpha"],
        beta=joint_selection["beta"],
        num_classes=len(id_to_label),
    )

    blend_selection = tune_hierarchical_blend(
        val_features=val_bundle.features,
        val_targets=val_bundle.targets,
        adjusted_joint_logits=val_joint_adjusted,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale_grid=parse_float_grid(args.crop_scale_grid),
        blend_grid=parse_float_grid(args.blend_grid),
    )

    _, val_crop_log_probs = predict_crop_from_joint_scores(val_joint_adjusted, crop_to_ids)
    _, test_crop_log_probs = predict_crop_from_joint_scores(test_joint_adjusted, crop_to_ids)
    val_hier_scores = build_hierarchical_joint_scores(
        features=val_bundle.features,
        crop_log_probs=val_crop_log_probs,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale=blend_selection.crop_scale,
    )
    test_hier_scores = build_hierarchical_joint_scores(
        features=test_bundle.features,
        crop_log_probs=test_crop_log_probs,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale=blend_selection.crop_scale,
    )

    val_final_scores = blend_selection.blend_weight * log_softmax_numpy(val_joint_adjusted) + (1.0 - blend_selection.blend_weight) * log_softmax_numpy(val_hier_scores)
    test_final_scores = blend_selection.blend_weight * log_softmax_numpy(test_joint_adjusted) + (1.0 - blend_selection.blend_weight) * log_softmax_numpy(test_hier_scores)

    baseline_metrics = evaluate_scores(log_softmax_numpy(test_bundle.logits), test_bundle.targets, id_to_label)
    final_metrics = evaluate_scores(test_final_scores, test_bundle.targets, id_to_label)

    summary = {
        "config": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "image_size": args.image_size,
            "c_grid": parse_float_grid(args.c_grid),
            "crop_scale_grid": parse_float_grid(args.crop_scale_grid),
            "blend_grid": parse_float_grid(args.blend_grid),
            "joint_alpha_grid": parse_float_grid(args.joint_alpha_grid),
            "joint_beta_grid": parse_float_grid(args.joint_beta_grid),
        },
        "cleanup_report": asdict(cleanup_report),
        "cleanup_weight_report": asdict(cleanup_weight_report),
        "joint_selection": joint_selection,
        "blend_selection": asdict(blend_selection),
        "baseline_test_metrics": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
            "weighted_f1": baseline_metrics["weighted_f1"],
        },
        "final_test_metrics": {
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "weighted_f1": final_metrics["weighted_f1"],
        },
        "per_crop_models": [asdict(item) for item in crop_models],
    }

    (args.output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "accuracy": baseline_metrics["accuracy"],
                    "macro_f1": baseline_metrics["macro_f1"],
                    "micro_f1": baseline_metrics["micro_f1"],
                    "weighted_f1": baseline_metrics["weighted_f1"],
                },
                "final": {
                    "accuracy": final_metrics["accuracy"],
                    "macro_f1": final_metrics["macro_f1"],
                    "micro_f1": final_metrics["micro_f1"],
                    "weighted_f1": final_metrics["weighted_f1"],
                    "classification_report": final_metrics["classification_report"],
                },
            },
            indent=2,
        )
    )
    pd.DataFrame(
        final_metrics["confusion_matrix"],
        index=[id_to_label[i] for i in sorted(id_to_label)],
        columns=[id_to_label[i] for i in sorted(id_to_label)],
    ).to_csv(args.output_dir / "test_confusion_matrix.csv")

    print(json.dumps(summary, indent=2))
    print(f"saved: {args.output_dir / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
