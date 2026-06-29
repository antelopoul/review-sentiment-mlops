# Review Sentiment MLOps

Low-latency sentiment classification service for product reviews. Fine-tuned MiniLM exported to INT8 ONNX, served via FastAPI with batch inference and Prometheus metrics.

**Labels:** `negative` · `neutral` · `positive`

---

## Architecture

```
Client
  │
  ▼
FastAPI  (app/main.py)
  │   ├── /predict        single text
  │   ├── /predict/batch  up to 64 texts, one ONNX call
  │   ├── /health
  │   └── /metrics        Prometheus
  ▼
ONNX Runtime  (INT8 quantized MiniLM)
  │
  ▼
Prometheus metrics  (app/metrics.py)
  ├── request latency histogram
  ├── p95 / p99 rolling gauges (last 1000 requests)
  ├── predictions by label
  └── confidence distribution
```

Training uses PyTorch. Inference uses ONNX Runtime — no PyTorch overhead at serve time.

---

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI + ONNX inference server
│   ├── schema.py        # Versioned Pydantic schemas (V1, V2 …)
│   ├── metrics.py       # Prometheus metric definitions
│   └── model.py         # Training-time classifier (not used at serve time)
│
├── training/
│   ├── train.py         # Fine-tuning script
│   └── preprocess.py    # Sentence splitting + label mapping
│
├── onnx_model/
│   ├── model.onnx       # Full precision export
│   └── model.int8.onnx  # INT8 quantized (used in production)
│
├── models/sentimentv1/  # Tokenizer files
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Cloning the repository

The ONNX models (`onnx_model/`) and tokenizer files (`models/sentimentv1/`) are stored in [Git LFS](https://git-lfs.com). You need Git LFS installed before cloning, otherwise those files will be empty pointer stubs.

**1. Install Git LFS** (once per machine):

```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt install git-lfs

# Windows (winget)
winget install GitHub.GitLFS
```

Then register the LFS hooks with Git:

```bash
git lfs install
```

**2. Clone the repo** — LFS files are downloaded automatically:

```bash
git clone https://github.com/<your-org>/review-sentiment-mlops.git
cd review-sentiment-mlops
```

**3. Verify the artifacts are real files** (not pointer stubs):

```bash
git lfs ls-files
# Should list:
#   onnx_model/model.onnx
#   onnx_model/model.int8.onnx
#   models/sentimentv1/...
```

If you already cloned without LFS, pull the real files now:

```bash
git lfs pull
```

---

## Prerequisites

- Python 3.10–3.11
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git LFS (see above) — required to download the ONNX models and tokenizer

---

## Installation

```bash
uv sync
```

For training dependencies (PyTorch, datasets, etc.):

```bash
uv sync --group dev
```

---

## Running locally

### 1. Set the API key

The server requires an `X-API-Key` header on every prediction request. Set a fixed key so it does not regenerate on restart:

```bash
# PowerShell
$env:API_KEY = "mykey"

# bash / zsh
export API_KEY=mykey
```

Or create a `.env` file (picked up automatically by docker-compose):

```
API_KEY=mykey
```

If `API_KEY` is not set the server auto-generates one and prints it to the log on startup — copy it before making requests.

### 2. Start the server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

With live reload during development:

```bash
uv run uvicorn app.main:app --port 8000 --reload
```

### 3. Verify

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "./onnx_model/model.int8.onnx", "tokenizer": "./models/sentimentv1"}
```

---

## API reference

All prediction endpoints require the header:

```
X-API-Key: <your key>
```

### POST /predict — single text

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "X-API-Key: mykey" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

Request:

```json
{
  "schema_version": "sentiment_request_v1",
  "text": "This product is absolutely amazing!"
}
```

Response:

```json
{
  "schema_version": "sentiment_response_v1",
  "prediction": "positive",
  "label": 2,
  "confidence": 0.978,
  "logits": [-3.1, -1.4, 4.2],
  "model_version": "1.0.0",
  "request_id": "a3f9e1b2"
}
```

`label` mapping: `0` → negative, `1` → neutral, `2` → positive.

---

### POST /predict/batch — multiple texts (recommended for automation)

Up to 64 texts per call. All texts are tokenized together and processed in a **single ONNX forward pass** — dramatically lower per-item latency than calling `/predict` in a loop.

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "X-API-Key: mykey" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great quality, fast shipping!",
      "Broke after two days, very disappointing.",
      "It is okay, nothing special."
    ]
  }'
```

Request:

```json
{
  "schema_version": "sentiment_batch_request_v1",
  "texts": ["Great quality!", "Broke after two days.", "It is okay."]
}
```

Response:

```json
{
  "schema_version": "sentiment_batch_response_v1",
  "results": [
    {"prediction": "positive", "label": 2, "confidence": 0.96, "logits": [...], "model_version": "1.0.0", "request_id": "a1b2"},
    {"prediction": "negative", "label": 0, "confidence": 0.91, "logits": [...], "model_version": "1.0.0", "request_id": "c3d4"},
    {"prediction": "neutral",  "label": 1, "confidence": 0.74, "logits": [...], "model_version": "1.0.0", "request_id": "e5f6"}
  ],
  "model_version": "1.0.0",
  "request_id": "batch-a1b2c3"
}
```

---

### GET /health

No auth required. Used by load balancers and container health checks.

### GET /metrics

Prometheus text format. Scrape with Prometheus or inspect manually:

```bash
curl http://localhost:8000/metrics
```

Key metrics:

| Metric | Type | Description |
|---|---|---|
| `sentiment_request_latency_seconds` | Histogram | End-to-end request latency |
| `sentiment_p95_latency_seconds` | Gauge | Rolling p95 over last 1000 requests |
| `sentiment_p99_latency_seconds` | Gauge | Rolling p99 over last 1000 requests |
| `sentiment_requests_total` | Counter | Total requests by model and status |
| `sentiment_predictions_total` | Counter | Prediction counts by label |
| `sentiment_prediction_confidence` | Histogram | Confidence score distribution |
| `sentiment_errors_total` | Counter | Errors by exception type |
| `sentiment_invalid_input_total` | Counter | Rejected requests (empty / too long) |

---

## Interactive docs (Swagger UI)

Open [http://localhost:8000/docs](http://localhost:8000/docs) in the browser.

Click **Authorize** (top right) → enter your raw API key value (e.g. `mykey`) → click Authorize → Close. All subsequent requests from the UI will include the header.

---

## Single vs batch — when to use which

| Scenario | Endpoint | Why |
|---|---|---|
| User submits a review on a website | `/predict` | Real-time, result needed immediately |
| Nightly job classifying new reviews from DB | `/predict/batch` | One round-trip for N reviews |
| Historical backfill of 1M reviews | `/predict/batch` in chunks | Maximises ONNX throughput |
| Kafka consumer processing review events | `/predict/batch` with micro-batching | Amortise network + session overhead |

For automated review segment analysis, chunk your reviews into groups of **32–64** (the `MAX_BATCH_SIZE` limit) and send them in parallel workers. A single batch call for 64 reviews takes roughly the same time as 3–4 single calls.

---

## Running with Docker

### Build and run

```bash
docker compose up --build
```

The service starts on port `8000`. Set the API key in your `.env` file before starting.

### Load test (optional)

```bash
docker compose --profile loadtest up
```

Opens the Locust UI at [http://localhost:8089](http://localhost:8089).

### Scaling workers

Control the number of gunicorn workers via the env var (default 4):

```bash
WEB_CONCURRENCY=8 docker compose up
```

---

## Cloud deployment

The container is stateless — mount or bake in the model artifacts and set `API_KEY` as a secret.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | auto-generated | Auth key for prediction endpoints |
| `MODEL_PATH` | `./onnx_model/model.int8.onnx` | Path to ONNX model file |
| `TOKENIZER_PATH` | `./models/sentimentv1` | Path to tokenizer directory |
| `WEB_CONCURRENCY` | `4` | Number of gunicorn workers |

### Platforms

- **AWS ECS / Fargate** — push image to ECR, set env vars as secrets in the task definition
- **Google Cloud Run** — `gcloud run deploy`, set `API_KEY` via Secret Manager
- **Azure Container Apps** — set secrets in the container app environment
- **Kubernetes (EKS / GKE / AKS)** — use a `Deployment` with a `Secret` for `API_KEY`; the `/health` endpoint is ready for liveness and readiness probes

### Kubernetes probe example

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 15
```

---

## Training

Fine-tune the model from the repository root:

```bash
uv run python -m training.train
```

Export to ONNX and quantize after training:

```bash
uv run python export_quantize_onnx.py
```

---

## Performance targets (CPU)

| Percentile | Target | Model |
|---|---|---|
| p50 | ~20–50 ms | INT8 ONNX, batch=1 |
| p95 | ~50–120 ms | INT8 ONNX, batch=1 |
| p99 | < 300 ms | INT8 ONNX, batch=1 |

Batch inference (32–64 texts) achieves ~5–8× better throughput than equivalent single calls at similar or lower p99.
