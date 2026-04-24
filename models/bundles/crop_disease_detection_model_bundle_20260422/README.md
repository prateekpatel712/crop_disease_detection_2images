# Crop Disease Detection Model Bundle

This bundle contains the currently selected hierarchical crop-disease model snapshot from `/home/jupyter`.

## What Is Inside

- `models/best_model.pt`
  - EfficientNet-B0 backbone checkpoint used for feature extraction.
- `models/crop_router_refinement.joblib`
  - Crop router trained on extracted features.
- `models/per_crop_heads/*.joblib`
  - One disease head per crop.
- `metadata/id_to_label.json`
  - Authoritative class-id to label mapping for the final 134 classes.
- `metadata/label_to_id.json`
  - Reverse mapping.
- `metadata/crop_order.json`
  - Crop ordering used by the router and score assembly.
- `metadata/preprocessing_config.json`
  - Image preprocessing used before inference.
- `metadata/pipeline_summary.json`
  - Stage-1 hierarchical training summary.
- `metadata/crop_router_refinement_summary.json`
  - Final router refinement summary for the selected model.
- `metadata/bundle_manifest.json`
  - Compact description of the bundle layout and selected inference settings.
- `scripts/professional_hierarchical_crop_disease_pipeline.py`
  - Original stage-1 pipeline.
- `scripts/professional_crop_router_refinement.py`
  - Original crop-router refinement pipeline.
- `scripts/predict_bundle_reference.py`
  - Small reference inference script for this bundle.

## Current Inference Flow

1. Load image as RGB.
2. Resize shorter side to `255`.
3. Center crop to `224 x 224`.
4. Normalize with ImageNet mean/std.
5. Run EfficientNet-B0 once to get the penultimate feature vector.
6. Run the crop router on the feature vector to get crop log-probabilities.
7. Run each crop-specific disease head on the same feature vector.
8. Combine crop score and disease-within-crop score using `crop_scale = 0.4`.
9. Apply softmax over the final 134-class scores.
10. Map predicted class id through `metadata/id_to_label.json`.

For this saved refined model, the final selected `blend_weight` is `0.0`, so the delivered predictor uses the hierarchical path directly from router + per-crop heads.

## Order Conventions

- Crop order is the sorted crop list in `metadata/crop_order.json`.
- Class order is ascending integer key order from `metadata/id_to_label.json`.

## Reference Usage

```bash
python scripts/predict_bundle_reference.py --image /path/to/image.jpg
```

For top-k output:

```bash
python scripts/predict_bundle_reference.py --image /path/to/image.jpg --topk 5
```
