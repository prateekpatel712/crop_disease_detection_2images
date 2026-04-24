# Crop Model Slot

Add trained crop-classification model artifacts here in a later phase.

Recommended future contents:

- model weights such as `.pt`, `.pth`, or `.onnx`
- label map files such as `labels.txt` or `labels.json`
- preprocessing metadata

Current registered artifact:

- `best_model.pt` as the first crop-classification checkpoint
- `model.json` with source and inspection metadata

Phase 6 runtime notes:

- the backend now includes a torchvision-based classifier runtime for this slot
- named predictions still require a real `labels.txt` or `labels.json`
- the current runtime defaults assume an ImageNet-style EfficientNet workflow and should be validated against the original training code
