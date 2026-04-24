param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectId,

    [Parameter(Mandatory = $true)]
    [string]$Region,

    [string]$ServiceName = "crop-disease-detection-api",
    [string]$Repository = "crop-disease-detection",
    [string]$ApiSecretName = "crop-disease-api-key"
)

$ErrorActionPreference = "Stop"

$image = "$Region-docker.pkg.dev/$ProjectId/$Repository/$ServiceName`:latest"

Write-Host "Setting gcloud project to $ProjectId..." -ForegroundColor Cyan
gcloud config set project $ProjectId | Out-Null

Write-Host "Enabling required Google Cloud APIs..." -ForegroundColor Cyan
gcloud services enable `
    run.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    secretmanager.googleapis.com | Out-Null

Write-Host "Ensuring Artifact Registry repository '$Repository' exists..." -ForegroundColor Cyan
try {
    gcloud artifacts repositories describe $Repository --location=$Region | Out-Null
}
catch {
    gcloud artifacts repositories create $Repository `
        --repository-format=docker `
        --location=$Region `
        --description="Container images for crop disease detection" | Out-Null
}

Write-Host "Building container image $image ..." -ForegroundColor Cyan
gcloud builds submit --tag $image .

Write-Host "Checking Secret Manager secret '$ApiSecretName'..." -ForegroundColor Cyan
try {
    gcloud secrets describe $ApiSecretName | Out-Null
}
catch {
    throw "Secret '$ApiSecretName' does not exist. Create it first with: gcloud secrets create $ApiSecretName --replication-policy=automatic"
}

Write-Host "Deploying Cloud Run service '$ServiceName'..." -ForegroundColor Cyan
gcloud run deploy $ServiceName `
    --image $image `
    --region $Region `
    --platform managed `
    --port 8080 `
    --cpu 1 `
    --memory 4Gi `
    --concurrency 1 `
    --timeout 120 `
    --min-instances 0 `
    --max-instances 1 `
    --allow-unauthenticated `
    --set-env-vars "APP_ENV=production,APP_DATA_DIR=/tmp/crop-disease-detection,ENABLE_API_DOCS=false,REQUIRE_API_KEY=true" `
    --set-secrets "API_KEY=$ApiSecretName:latest"

Write-Host ""
Write-Host "Deployment complete." -ForegroundColor Green
Write-Host "Make sure the secret has at least one version:" -ForegroundColor Yellow
Write-Host "  Write-Output 'your-long-random-api-key' | gcloud secrets versions add $ApiSecretName --data-file=-"
