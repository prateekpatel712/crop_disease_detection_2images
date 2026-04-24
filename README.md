# Crop Disease Detection API

Two-image crop disease detection API with:

- crop identification from a reference crop leaf image
- disease prediction from a diseased leaf image
- RAG-backed disease guidance
- API key protection via `X-API-Key`
- Swagger/OpenAPI disabled by default in production

## What is included

- Full hierarchical model bundle under `models/bundles/crop_disease_detection_model_bundle_20260422/`
- Processed RAG corpus in `rag_data/processed/disease_records.jsonl`
- Cloud Run deployment files:
  - `Dockerfile`
  - `.dockerignore`
  - `.gcloudignore`
  - `deploy_cloud_run.ps1`

## Local configuration

Copy `.env.example` to `.env` and set a real API key.

Example:

```env
APP_ENV=production
APP_DATA_DIR=/tmp/crop-disease-detection
API_KEY=replace-with-a-long-random-secret
REQUIRE_API_KEY=true
ENABLE_API_DOCS=false
```

## Local run

```powershell
python -m app.scripts.build_rag_index
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Send the API key as `X-API-Key`.

## Cloud Run deployment

Prerequisites:

- Google Cloud project with billing enabled
- `gcloud` installed and authenticated
- a Secret Manager secret for the API key

Create the secret:

```powershell
gcloud secrets create crop-disease-api-key --replication-policy=automatic
Write-Output 'your-long-random-api-key' | gcloud secrets versions add crop-disease-api-key --data-file=-
```

Deploy:

```powershell
.\deploy_cloud_run.ps1 -ProjectId YOUR_PROJECT_ID -Region asia-south1
```

The deployment script will:

- enable required APIs
- create Artifact Registry if needed
- build the container image
- deploy Cloud Run with:
  - `1` CPU
  - `4Gi` memory
  - concurrency `1`
  - min instances `0`
  - max instances `1`
  - docs disabled
  - `APP_DATA_DIR=/tmp/crop-disease-detection`

## Request format

`POST /api/v1/predict`

Multipart form fields:

- `crop_image`
- `diseased_image`

Header:

- `X-API-Key: your-secret`

## Notes

- Uploaded images, logs, and generated vector indices are runtime data and are not persisted by default on Cloud Run.
- The app rebuilds the RAG index at container startup.
- For hackathon/demo use, this setup is fine. For long-term production, move uploads/logs to managed storage.
