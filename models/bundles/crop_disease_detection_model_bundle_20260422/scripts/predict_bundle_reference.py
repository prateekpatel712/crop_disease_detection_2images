#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reference predictor for the crop disease model bundle.")
    parser.add_argument("--image", type=Path, required=True, help="Path to one input image.")
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Bundle root directory. Defaults to the parent of this script.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of predictions to print.")
    return parser.parse_args()


class EfficientNetFeatureModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.30, inplace=True),
            torch.nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.model.features(x)
        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.model.classifier(features)
        return features, logits


def build_transform(config: dict) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(config["resize_shorter_side"])),
            transforms.CenterCrop(int(config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(config["normalize_mean"], config["normalize_std"]),
        ]
    )


def load_image_tensor(image_path: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(image_path) as img:
        return transform(img.convert("RGB")).unsqueeze(0)


def load_id_to_label(path: Path) -> Dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text()).items()}


def build_crop_to_ids(id_to_label: Dict[int, str]) -> Dict[str, List[int]]:
    crop_to_ids: Dict[str, List[int]] = {}
    for class_id, label in sorted(id_to_label.items()):
        crop = label.split("::", 1)[0]
        crop_to_ids.setdefault(crop, []).append(class_id)
    return crop_to_ids


def load_crop_model_payloads(model_dir: Path) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for path in sorted(model_dir.glob("*_disease_head.joblib")):
        crop_name = path.name.replace("_disease_head.joblib", "")
        payloads[crop_name] = joblib.load(path)
    return payloads


def predict_crop_router_log_probs(features: np.ndarray, payload: dict) -> np.ndarray:
    x_eval = features
    scaler = payload.get("scaler")
    if scaler is not None:
        x_eval = scaler.transform(x_eval)
    classifier: LogisticRegression = payload["classifier"]
    return classifier.predict_log_proba(x_eval)


def build_hierarchical_scores(
    features: np.ndarray,
    crop_log_probs: np.ndarray,
    crop_to_ids: Dict[str, List[int]],
    crop_model_payloads: Dict[str, dict],
    crop_scale: float,
) -> np.ndarray:
    crop_names = sorted(crop_to_ids)
    crop_name_to_col = {crop: idx for idx, crop in enumerate(crop_names)}
    num_classes = sum(len(v) for v in crop_to_ids.values())
    scores = np.full((features.shape[0], num_classes), -1e9, dtype=np.float64)

    for crop, class_ids in crop_to_ids.items():
        payload = crop_model_payloads[crop]
        if payload.get("kind") == "constant":
            disease_log_probs = np.full((features.shape[0], len(class_ids)), -1e9, dtype=np.float64)
            disease_log_probs[:, int(payload["constant_local_class"])] = 0.0
        else:
            scaler: StandardScaler = payload["scaler"]
            classifier: LogisticRegression = payload["classifier"]
            x_scaled = scaler.transform(features)
            raw_log_probs = classifier.predict_log_proba(x_scaled)
            disease_log_probs = np.full((features.shape[0], len(class_ids)), -1e9, dtype=np.float64)
            seen_local_classes = payload.get("seen_local_classes", list(range(raw_log_probs.shape[1])))
            for col_idx, local_class_idx in enumerate(seen_local_classes):
                disease_log_probs[:, int(local_class_idx)] = raw_log_probs[:, col_idx]

        crop_col = crop_name_to_col[crop]
        for local_idx, class_id in enumerate(class_ids):
            scores[:, class_id] = crop_scale * crop_log_probs[:, crop_col] + disease_log_probs[:, local_idx]
    return scores


def main() -> None:
    args = parse_args()
    bundle_root = args.bundle_root.resolve()
    metadata_dir = bundle_root / "metadata"
    models_dir = bundle_root / "models"

    preprocess_config = json.loads((metadata_dir / "preprocessing_config.json").read_text())
    refinement_summary = json.loads((metadata_dir / "crop_router_refinement_summary.json").read_text())
    id_to_label = load_id_to_label(metadata_dir / "id_to_label.json")
    crop_to_ids = build_crop_to_ids(id_to_label)

    model = EfficientNetFeatureModel(num_classes=len(id_to_label))
    checkpoint = torch.load(models_dir / "best_model.pt", map_location="cpu")
    model.model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transform(preprocess_config)
    image_tensor = load_image_tensor(args.image, transform)

    with torch.no_grad():
        features_tensor, _ = model(image_tensor)
    features = features_tensor.cpu().numpy()

    router_payload = joblib.load(models_dir / "crop_router_refinement.joblib")
    crop_model_payloads = load_crop_model_payloads(models_dir / "per_crop_heads")
    crop_log_probs = predict_crop_router_log_probs(features, router_payload)

    selected = refinement_summary["crop_router_selection"]
    crop_scale = float(selected["crop_scale"])
    blend_weight = float(selected["blend_weight"])
    if blend_weight != 0.0:
        raise RuntimeError(
            f"This reference script expects blend_weight 0.0 for the bundled model, found {blend_weight}."
        )

    final_scores = build_hierarchical_scores(
        features=features,
        crop_log_probs=crop_log_probs,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale=crop_scale,
    )
    probabilities = torch.softmax(torch.from_numpy(final_scores), dim=1).numpy()[0]

    topk = max(1, min(int(args.topk), len(id_to_label)))
    top_indices = np.argsort(probabilities)[::-1][:topk]

    print(f"image={args.image}")
    for rank, class_id in enumerate(top_indices.tolist(), start=1):
        label = id_to_label[int(class_id)]
        probability = float(probabilities[int(class_id)])
        print(f"{rank}. class_id={class_id} label={label} probability={probability:.6f}")


if __name__ == "__main__":
    main()
