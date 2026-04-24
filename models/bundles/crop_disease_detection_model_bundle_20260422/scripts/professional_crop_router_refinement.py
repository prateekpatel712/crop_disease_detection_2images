from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import professional_hierarchical_crop_disease_pipeline as base


DEFAULT_OUTPUT_DIR = Path("/home/jupyter/combined_plant_dataset_normalized/professional_hierarchical_outputs")
DEFAULT_PREPARED_DIR = Path(
    "/home/jupyter/combined_plant_dataset_normalized/professional_all_crop_disease_outputs/prepared_data"
)


@dataclass
class CropRouterSelection:
    standardize: bool
    class_weight: str
    selected_c: float
    crop_scale: float
    blend_weight: float
    val_accuracy: float
    val_macro_f1: float
    crop_val_accuracy: float
    crop_val_macro_f1: float
    model_path: str


def parse_bool_grid(values: str) -> List[bool]:
    parsed: List[bool] = []
    for token in values.split(","):
        token = token.strip().lower()
        if token in {"1", "true", "yes", "y"}:
            parsed.append(True)
        elif token in {"0", "false", "no", "n"}:
            parsed.append(False)
        else:
            raise ValueError(f"Unsupported boolean token: {token}")
    return parsed


def parse_string_grid(values: str) -> List[str]:
    return [token.strip() for token in values.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine the hierarchical crop-disease system with a dedicated crop router on cached features."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--router-c-grid", type=str, default="0.03,0.1,0.3,1.0,3.0,10.0")
    parser.add_argument("--router-scale-grid", type=str, default="0.25,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument("--blend-grid", type=str, default="0.0,0.05,0.1,0.15,0.2,0.25")
    parser.add_argument("--router-standardize-grid", type=str, default="true,false")
    parser.add_argument("--router-class-weight-grid", type=str, default="none,balanced")
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


def load_feature_bundle(path: Path) -> base.FeatureBundle:
    payload = torch.load(path, map_location="cpu")
    return base.FeatureBundle(
        features=payload["features"].numpy(),
        logits=payload["logits"].numpy(),
        targets=payload["targets"].numpy(),
    )


def load_crop_model_payloads(model_dir: Path) -> Dict[str, dict]:
    payloads: Dict[str, dict] = {}
    for path in model_dir.glob("*_disease_head.joblib"):
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


def fit_and_select_crop_router(
    train_df: pd.DataFrame,
    train_features: np.ndarray,
    val_features: np.ndarray,
    val_crop_y: np.ndarray,
    val_targets: np.ndarray,
    crop_to_ids: Dict[str, List[int]],
    crop_model_payloads: Dict[str, dict],
    joint_val_log_probs: np.ndarray,
    output_path: Path,
    router_c_grid: Iterable[float],
    router_scale_grid: Iterable[float],
    blend_grid: Iterable[float],
    standardize_grid: Iterable[bool],
    class_weight_grid: Iterable[str],
    max_iter: int,
) -> tuple[CropRouterSelection, dict]:
    crop_names = sorted(crop_to_ids)
    crop_to_idx = {crop: idx for idx, crop in enumerate(crop_names)}
    train_crop_y = train_df["crop_label"].map(crop_to_idx).astype(int).to_numpy()

    best_selection: Optional[CropRouterSelection] = None
    best_payload: Optional[dict] = None
    best_score: Optional[float] = None

    for standardize in standardize_grid:
        if standardize:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(train_features)
            x_val = scaler.transform(val_features)
        else:
            scaler = None
            x_train = train_features
            x_val = val_features

        for class_weight_name in class_weight_grid:
            class_weight = None if class_weight_name == "none" else class_weight_name
            for c_value in router_c_grid:
                classifier = LogisticRegression(
                    C=float(c_value),
                    max_iter=max_iter,
                    solver="lbfgs",
                    class_weight=class_weight,
                )
                classifier.fit(x_train, train_crop_y)

                val_crop_pred = classifier.predict(x_val)
                crop_val_accuracy = float(np.mean(val_crop_pred == val_crop_y))
                crop_val_macro_f1 = float(f1_score(val_crop_y, val_crop_pred, average="macro"))
                val_crop_log_probs = classifier.predict_log_proba(x_val)

                for crop_scale in router_scale_grid:
                    val_hier_scores = base.build_hierarchical_joint_scores(
                        features=val_features,
                        crop_log_probs=val_crop_log_probs,
                        crop_to_ids=crop_to_ids,
                        crop_model_payloads=crop_model_payloads,
                        crop_scale=float(crop_scale),
                    )
                    val_hier_log_probs = base.log_softmax_numpy(val_hier_scores)

                    for blend_weight in blend_grid:
                        val_scores = (
                            float(blend_weight) * joint_val_log_probs
                            + (1.0 - float(blend_weight)) * val_hier_log_probs
                        )
                        val_preds = val_scores.argmax(axis=1)
                        val_accuracy = float(np.mean(val_preds == val_targets))
                        val_macro_f1 = float(f1_score(val_targets, val_preds, average="macro"))
                        selection_score = base.weighted_selection_score(val_macro_f1, val_accuracy)

                        if best_score is None or selection_score > best_score:
                            best_score = selection_score
                            best_payload = {
                                "scaler": scaler,
                                "classifier": classifier,
                                "crop_names": crop_names,
                                "standardize": bool(standardize),
                                "class_weight": class_weight_name,
                                "selected_c": float(c_value),
                                "crop_scale": float(crop_scale),
                                "blend_weight": float(blend_weight),
                            }
                            best_selection = CropRouterSelection(
                                standardize=bool(standardize),
                                class_weight=class_weight_name,
                                selected_c=float(c_value),
                                crop_scale=float(crop_scale),
                                blend_weight=float(blend_weight),
                                val_accuracy=val_accuracy,
                                val_macro_f1=val_macro_f1,
                                crop_val_accuracy=crop_val_accuracy,
                                crop_val_macro_f1=crop_val_macro_f1,
                                model_path=str(output_path),
                            )

    if best_selection is None or best_payload is None:
        raise RuntimeError("No crop router candidate was selected.")

    joblib.dump(best_payload, output_path)
    return best_selection, best_payload


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    prepared_dir = args.prepared_dir
    feature_dir = output_dir / "feature_cache"
    model_dir = output_dir / "per_crop_models"
    router_path = output_dir / "crop_router_refinement.joblib"

    label_to_id = json.load(open(prepared_dir / "label_to_id.json"))
    id_to_label = {int(k): v for k, v in json.load(open(prepared_dir / "id_to_label.json")).items()}
    _, crop_to_ids = base.build_label_maps(id_to_label)

    train_df = pd.read_csv(prepared_dir / "train_manifest.csv", low_memory=False)
    val_df = pd.read_csv(prepared_dir / "val_manifest.csv", low_memory=False)
    test_df = pd.read_csv(prepared_dir / "test_manifest.csv", low_memory=False)
    train_df, val_df, test_df, cleanup_report = base.cleanup_manifests(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_to_id=label_to_id,
        majority_threshold=0.8,
    )

    train_bundle = load_feature_bundle(feature_dir / "train_features.pt")
    val_bundle = load_feature_bundle(feature_dir / "val_features.pt")
    test_bundle = load_feature_bundle(feature_dir / "test_features.pt")
    crop_model_payloads = load_crop_model_payloads(model_dir)

    if len(crop_model_payloads) != len(crop_to_ids):
        raise RuntimeError(
            f"Expected {len(crop_to_ids)} crop heads, found {len(crop_model_payloads)} in {model_dir}."
        )

    prior_summary = json.load(open(output_dir / "pipeline_summary.json"))
    joint_selection = prior_summary["joint_selection"]
    val_joint_adjusted = base.apply_joint_prior_adjustment(
        logits=val_bundle.logits,
        train_df=train_df,
        crop_to_ids=crop_to_ids,
        alpha=float(joint_selection["alpha"]),
        beta=float(joint_selection["beta"]),
        num_classes=len(id_to_label),
    )
    test_joint_adjusted = base.apply_joint_prior_adjustment(
        logits=test_bundle.logits,
        train_df=train_df,
        crop_to_ids=crop_to_ids,
        alpha=float(joint_selection["alpha"]),
        beta=float(joint_selection["beta"]),
        num_classes=len(id_to_label),
    )
    val_joint_log_probs = base.log_softmax_numpy(val_joint_adjusted)
    test_joint_log_probs = base.log_softmax_numpy(test_joint_adjusted)

    crop_names = sorted(crop_to_ids)
    crop_to_idx = {crop: idx for idx, crop in enumerate(crop_names)}
    val_crop_y = val_df["crop_label"].map(crop_to_idx).astype(int).to_numpy()
    test_crop_y = test_df["crop_label"].map(crop_to_idx).astype(int).to_numpy()

    selection, router_payload = fit_and_select_crop_router(
        train_df=train_df,
        train_features=train_bundle.features,
        val_features=val_bundle.features,
        val_crop_y=val_crop_y,
        val_targets=val_bundle.targets,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        joint_val_log_probs=val_joint_log_probs,
        output_path=router_path,
        router_c_grid=base.parse_float_grid(args.router_c_grid),
        router_scale_grid=base.parse_float_grid(args.router_scale_grid),
        blend_grid=base.parse_float_grid(args.blend_grid),
        standardize_grid=parse_bool_grid(args.router_standardize_grid),
        class_weight_grid=parse_string_grid(args.router_class_weight_grid),
        max_iter=args.max_iter,
    )

    val_crop_log_probs = predict_crop_router_log_probs(val_bundle.features, router_payload)
    test_crop_log_probs = predict_crop_router_log_probs(test_bundle.features, router_payload)

    val_crop_preds = val_crop_log_probs.argmax(axis=1)
    test_crop_preds = test_crop_log_probs.argmax(axis=1)

    val_hier_scores = base.build_hierarchical_joint_scores(
        features=val_bundle.features,
        crop_log_probs=val_crop_log_probs,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale=selection.crop_scale,
    )
    test_hier_scores = base.build_hierarchical_joint_scores(
        features=test_bundle.features,
        crop_log_probs=test_crop_log_probs,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale=selection.crop_scale,
    )

    val_hier_log_probs = base.log_softmax_numpy(val_hier_scores)
    test_hier_log_probs = base.log_softmax_numpy(test_hier_scores)

    val_final_scores = (
        selection.blend_weight * val_joint_log_probs
        + (1.0 - selection.blend_weight) * val_hier_log_probs
    )
    test_final_scores = (
        selection.blend_weight * test_joint_log_probs
        + (1.0 - selection.blend_weight) * test_hier_log_probs
    )

    baseline_metrics = base.evaluate_scores(base.log_softmax_numpy(test_bundle.logits), test_bundle.targets, id_to_label)
    _, test_stage1_crop_log_probs = base.predict_crop_from_joint_scores(test_joint_adjusted, crop_to_ids)
    test_stage1_hier_scores = base.build_hierarchical_joint_scores(
        features=test_bundle.features,
        crop_log_probs=test_stage1_crop_log_probs,
        crop_to_ids=crop_to_ids,
        crop_model_payloads=crop_model_payloads,
        crop_scale=float(prior_summary["blend_selection"]["crop_scale"]),
    )
    test_stage1_scores = (
        float(prior_summary["blend_selection"]["blend_weight"]) * test_joint_log_probs
        + (1.0 - float(prior_summary["blend_selection"]["blend_weight"]))
        * base.log_softmax_numpy(test_stage1_hier_scores)
    )
    stage1_metrics = base.evaluate_scores(test_stage1_scores, test_bundle.targets, id_to_label)
    final_metrics = base.evaluate_scores(test_final_scores, test_bundle.targets, id_to_label)

    summary = {
        "cleanup_report": asdict(cleanup_report),
        "router_search_config": {
            "router_c_grid": base.parse_float_grid(args.router_c_grid),
            "router_scale_grid": base.parse_float_grid(args.router_scale_grid),
            "blend_grid": base.parse_float_grid(args.blend_grid),
            "router_standardize_grid": parse_bool_grid(args.router_standardize_grid),
            "router_class_weight_grid": parse_string_grid(args.router_class_weight_grid),
            "max_iter": args.max_iter,
        },
        "prior_stage_joint_selection": joint_selection,
        "prior_stage_hierarchical_test_accuracy": float(prior_summary["final_test_metrics"]["accuracy"]),
        "crop_router_selection": asdict(selection),
        "crop_router_validation_metrics": {
            "accuracy": float(np.mean(val_crop_preds == val_crop_y)),
            "macro_f1": float(f1_score(val_crop_y, val_crop_preds, average="macro")),
        },
        "crop_router_test_metrics": {
            "accuracy": float(np.mean(test_crop_preds == test_crop_y)),
            "macro_f1": float(f1_score(test_crop_y, test_crop_preds, average="macro")),
        },
        "baseline_test_metrics": {
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
            "weighted_f1": baseline_metrics["weighted_f1"],
        },
        "stage1_hierarchical_test_metrics": {
            "accuracy": stage1_metrics["accuracy"],
            "macro_f1": stage1_metrics["macro_f1"],
            "weighted_f1": stage1_metrics["weighted_f1"],
        },
        "final_test_metrics": {
            "accuracy": final_metrics["accuracy"],
            "macro_f1": final_metrics["macro_f1"],
            "weighted_f1": final_metrics["weighted_f1"],
        },
    }

    (output_dir / "crop_router_refinement_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "crop_router_refined_test_metrics.json").write_text(
        json.dumps(
            {
                "baseline": {
                    "accuracy": baseline_metrics["accuracy"],
                    "macro_f1": baseline_metrics["macro_f1"],
                    "micro_f1": baseline_metrics["micro_f1"],
                    "weighted_f1": baseline_metrics["weighted_f1"],
                },
                "stage1_hierarchical": {
                    "accuracy": stage1_metrics["accuracy"],
                    "macro_f1": stage1_metrics["macro_f1"],
                    "micro_f1": stage1_metrics["micro_f1"],
                    "weighted_f1": stage1_metrics["weighted_f1"],
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
    ).to_csv(output_dir / "crop_router_refined_confusion_matrix.csv")

    print(json.dumps(summary, indent=2))
    print(f"saved: {output_dir / 'crop_router_refinement_summary.json'}")


if __name__ == "__main__":
    main()
