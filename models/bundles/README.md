# Hierarchical Model Bundles

Place complete crop-disease model bundles here when the runtime depends on more than one artifact.

Current registered bundle:

- `crop_disease_detection_model_bundle_20260422/`

Expected bundle contents:

- `metadata/bundle_manifest.json`
- `metadata/id_to_label.json`
- `metadata/crop_order.json`
- `metadata/preprocessing_config.json`
- `models/best_model.pt`
- `models/crop_router_refinement.joblib`
- `models/per_crop_heads/*.joblib`

The backend will automatically discover the newest bundle that contains a valid `bundle_manifest.json`.
