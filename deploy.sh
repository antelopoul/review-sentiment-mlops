#!/usr/bin/env bash
# ============================================================
# Cloud Run deployment script — sentiment API
# Region: europe-west1
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
# ============================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. CONFIG — fill these in before running
# ---------------------------------------------------------------------------

PROJECT_ID="sentiment-project"          # e.g. "my-sentiment-project"
SERVICE_NAME="sentiment-api"
REGION="europe-west1"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
API_KEY=$(grep '^API_KEY=' .env | cut -d '=' -f2)

# ---------------------------------------------------------------------------
# 2. PREFLIGHT CHECKS
# ---------------------------------------------------------------------------

if [ -z "$PROJECT_ID" ]; then
  echo "ERROR: Set PROJECT_ID at the top of this script before running."
  exit 1
fi

if [ -z "$API_KEY" ]; then
  echo "ERROR: API_KEY not found in .env"
  exit 1
fi

if ! command -v gcloud &>/dev/null; then
  echo "ERROR: gcloud not found."
  echo "Install it from: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

echo "✓ Config OK — project=${PROJECT_ID}, region=${REGION}"

# ---------------------------------------------------------------------------
# 3. FIRST-TIME SETUP (safe to re-run — skips if already done)
# ---------------------------------------------------------------------------

echo ""
echo "==> Authenticating with Google Cloud..."
gcloud auth login --quiet

echo ""
echo "==> Creating project '${PROJECT_ID}' (skip if exists)..."
gcloud projects create "${PROJECT_ID}" --quiet 2>/dev/null || true

echo ""
echo "==> Setting active project..."
gcloud config set project "${PROJECT_ID}"

echo ""
echo ">>> ACTION REQUIRED: Enable billing for project '${PROJECT_ID}' at:"
echo "    https://console.cloud.google.com/billing/linkedaccount?project=${PROJECT_ID}"
echo ""
read -rp "Press ENTER once billing is enabled to continue..."

echo ""
echo "==> Enabling required APIs..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  containerregistry.googleapis.com \
  --quiet

echo ""
echo "==> Configuring Docker to authenticate with GCR..."
gcloud auth configure-docker --quiet

# ---------------------------------------------------------------------------
# 4. BUILD & PUSH IMAGE
# ---------------------------------------------------------------------------

echo ""
echo "==> Building and pushing Docker image to GCR..."
echo "    Image: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}" .

# ---------------------------------------------------------------------------
# 5. DEPLOY TO CLOUD RUN
# ---------------------------------------------------------------------------

echo ""
echo "==> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --port 8000 \
  --memory 1Gi \
  --cpu 2 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars "API_KEY=${API_KEY}" \
  --no-allow-unauthenticated \
  --quiet

# ---------------------------------------------------------------------------
# 6. DONE — print service URL
# ---------------------------------------------------------------------------

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format "value(status.url)")

echo ""
echo "============================================================"
echo "  Deployment complete!"
echo "  URL:  ${SERVICE_URL}"
echo "  Key:  ${API_KEY}"
echo ""
echo "  Test it:"
echo "  curl -X POST ${SERVICE_URL}/predict \\"
echo "    -H 'X-API-Key: ${API_KEY}' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"This product is really good!\"}'"
echo "============================================================"
