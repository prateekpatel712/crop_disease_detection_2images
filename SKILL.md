---
name: crop-disease-detection-project
description: Working brief for building an intelligent crop leaf classification, disease detection, and RAG explanation system with LangGraph orchestration.
---

# Crop Disease Detection Project Skill

Use this file as the project memory and implementation brief for this workspace.

## Project Goal

Build an intelligent system that:

1. Accepts a crop leaf image from the user.
2. Identifies which crop the leaf belongs to.
3. Detects whether the leaf is healthy or diseased.
4. Predicts the disease name if diseased.
5. Uses RAG to retrieve detailed information about that disease.
6. Returns a clear answer with disease details, precautions, prevention, and next steps.

The trained models and RAG store will be added later, so the codebase must be designed with clean placeholders and upload-ready directories.

## Core Product Principles

1. Do not force a prediction when the system is not confident.
2. Prefer safe refusal over a wrong diagnosis.
3. Keep model inference separate from RAG explanation.
4. Keep the workflow modular so models can be plugged in later.
5. Use LangGraph for orchestration and branching logic.
6. Design for both intelligence and speed.

## Required System Behavior

The system should support:

1. Image validation.
2. Image quality checks.
3. Leaf or non-leaf detection.
4. Healthy or diseased routing.
5. Crop classification.
6. Crop-specific disease routing.
7. Disease classification.
8. Confidence-based gating.
9. RAG retrieval for disease details.
10. Clear final explanation for the user.

## High-Level Flow

```text
User uploads image
-> validate_image
-> quality_check
-> detect_leaf
-> detect_healthy_or_diseased
-> classify_crop
-> crop_confidence_gate
-> route_disease_model
-> classify_disease
-> disease_confidence_gate
-> build_rag_query
-> retrieve_rag_docs
-> rerank_rag_docs
-> verify_prediction
-> generate_response
-> log_result
-> END
```

## Why LangGraph Is Used

LangGraph is used as the workflow brain, not as the model itself.

It is included because the system should support:

1. Conditional routing.
2. Confidence-based stopping.
3. Retry or fallback paths.
4. Different disease model selection per crop.
5. Early exits on bad images.
6. Later expansion into more agentic behavior.

## Target Folder Structure

```text
crop_disease_detection/
|- SKILL.md
|- app/
|  |- api/
|  |- core/
|  |- graph/
|  |- services/
|  |- models/
|  `- schemas/
|- models/
|  |- crop/
|  |- disease/
|  `- metadata/
|- rag_data/
|  |- raw/
|  |- processed/
|  `- vector_store/
|- uploads/
|- logs/
|- tests/
`- requirements.txt
```

## Reserved Integration Points

### Model Upload Slots

Keep dedicated paths for later model upload:

- `models/crop/`
- `models/disease/`
- `models/metadata/`

Expected future contents:

- model weights
- label maps
- preprocessing config
- model metadata

### RAG Upload Slots

Keep dedicated paths for later RAG integration:

- `rag_data/raw/`
- `rag_data/processed/`
- `rag_data/vector_store/`

Expected future contents:

- disease documents
- structured crop/disease entries
- chunked text
- embeddings/vector index

## Smart Features To Include

### Input Intelligence

1. File type validation.
2. File size validation.
3. Corrupt image detection.
4. Blur detection.
5. Brightness check.
6. Contrast check.
7. Resolution check.
8. Single-leaf or clutter detection if feasible.

### Prediction Intelligence

1. Leaf or non-leaf classifier.
2. Healthy or diseased classifier.
3. Crop classifier.
4. Crop confidence thresholding.
5. Top-3 crop candidates.
6. Crop-specific disease model routing.
7. Disease classifier.
8. Disease confidence thresholding.
9. Top-3 disease candidates.
10. `unknown` or `uncertain` fallback class.

### RAG Intelligence

1. Query built from crop name and disease name.
2. Metadata filtering by crop.
3. Retrieval from a vector store.
4. Re-ranking of retrieved chunks.
5. Fallback retrieval when disease-specific knowledge is weak.

### Verification Intelligence

1. Check whether predicted disease is valid for predicted crop.
2. Check whether retrieved symptoms match predicted disease.
3. Lower certainty if retrieval and prediction disagree.
4. Ask for another image when the result is weak.

### Performance Intelligence

1. Keep loaded models in memory.
2. Cache repeated RAG lookups.
3. Precompute embeddings.
4. Resize images before heavy inference.
5. Use async backend patterns where useful.

## Safety Rules

1. Never present a low-confidence result as certain.
2. Never force a disease answer for unsupported or poor-quality images.
3. Always allow `unknown`, `uncertain`, or `needs better image`.
4. RAG explains the prediction; it does not replace the disease classifier.
5. Return a cautionary message for severe or uncertain cases.
6. Encourage expert confirmation before pesticide or medical-grade action.

## Confidence Strategy

Use thresholds for both crop and disease prediction.

Recommended behavior:

1. High confidence:
   return primary result and continue.
2. Medium confidence:
   return result with caution and optional top alternatives.
3. Low confidence:
   stop hard diagnosis and ask for another image.

The exact threshold values should be configurable in code.

## State Design For LangGraph

Use a central graph state similar to:

```python
class PipelineState(TypedDict):
    image_path: str
    image_ok: bool
    quality_score: float
    quality_issues: list[str]
    is_leaf: bool
    is_healthy: bool | None
    crop_name: str
    crop_confidence: float
    crop_top_k: list[dict]
    disease_name: str
    disease_confidence: float
    disease_top_k: list[dict]
    rag_query: str
    rag_docs: list[dict]
    verified: bool
    warnings: list[str]
    final_answer: str
```
```

## Recommended LangGraph Nodes

Implement the workflow as small nodes:

1. `validate_image`
2. `quality_check`
3. `detect_leaf`
4. `detect_healthy_or_diseased`
5. `classify_crop`
6. `crop_confidence_gate`
7. `route_disease_model`
8. `classify_disease`
9. `disease_confidence_gate`
10. `build_rag_query`
11. `retrieve_rag_docs`
12. `rerank_rag_docs`
13. `verify_prediction`
14. `generate_response`
15. `log_result`

## Build Strategy

### Phase 1: Scaffold

1. Create backend structure.
2. Create LangGraph skeleton.
3. Add placeholder model services.
4. Add placeholder RAG service.
5. Add upload-ready folders.

### Phase 2: Smart Input Layer

1. Add validation and quality checks.
2. Add leaf detection placeholder.
3. Add healthy or diseased placeholder.

### Phase 3: Prediction Layer

1. Add crop prediction interface.
2. Add disease prediction interface.
3. Add confidence gates.
4. Add crop-specific routing.

### Phase 4: RAG Layer

1. Add disease document schema.
2. Add retrieval interface.
3. Add metadata filtering.
4. Add reranking.

### Phase 5: Verification and Output

1. Add prediction verification node.
2. Add response builder.
3. Add logging and feedback hooks.

### Phase 6: Real Model Integration

1. Plug in crop model.
2. Plug in disease model.
3. Plug in real vector store.
4. Tune thresholds using evaluation data.

## Placeholder Contract Design

All model and retrieval logic should be written behind interfaces.

Example contracts:

```python
def predict_crop(image) -> dict:
    return {
        "label": "unknown",
        "confidence": 0.0,
        "top_k": []
    }

def predict_disease(image, crop_name) -> dict:
    return {
        "label": "unknown",
        "confidence": 0.0,
        "top_k": []
    }

def retrieve_disease_details(crop_name, disease_name) -> list[dict]:
    return []
```
```

This allows the app to be built before training is complete.

## Data Design For RAG

Preferred knowledge structure:

```json
{
  "crop": "Tomato",
  "disease": "Early Blight",
  "summary": "Brief overview",
  "symptoms": ["..."],
  "causes": ["..."],
  "treatment": ["..."],
  "precautions": ["..."],
  "prevention": ["..."],
  "severity": "medium",
  "source": "reference"
}
```
```

This is better than only storing raw paragraphs.

## Output Expectations

The final user response should try to include:

1. Crop name.
2. Crop confidence.
3. Disease name.
4. Disease confidence.
5. Certainty or uncertainty status.
6. Symptoms.
7. Causes.
8. Precautions.
9. Prevention.
10. Suggested next step.

## What To Avoid

1. Do not tightly couple preprocessing logic to one backend route.
2. Do not hardcode model file names all over the codebase.
3. Do not let RAG decide the disease label.
4. Do not assume the model is always correct.
5. Do not build everything around a single monolithic script.

## Long-Term Improvements

Add these later if useful:

1. Region and season metadata.
2. Weather-aware reasoning.
3. User correction feedback loop.
4. Retraining data generation from reviewed cases.
5. Dashboard for low-confidence and misclassified cases.
6. ONNX or TensorRT optimization.
7. Mobile capture guidance for better images.

## Default Engineering Direction

When building in this workspace:

1. Prefer modular services.
2. Prefer configurable thresholds.
3. Prefer placeholder-friendly architecture.
4. Keep model integration points explicit.
5. Keep RAG integration points explicit.
6. Treat safety and confidence handling as first-class features.
