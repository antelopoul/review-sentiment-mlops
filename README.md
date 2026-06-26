# Sentiment Analysis Service (MiniLM + ONNX + FastAPI)

## Overview

This project implements a **low-latency sentiment classification service** that categorizes review sentences into:

* Positive
* Neutral
* Negative

The system is optimized for **production-style inference** with:

* P99 latency target < 300 ms
* Local and cloud deployment compatibility
* Full observability (logging + metrics)

---

## Architecture

```
Client
  │
  ▼
FastAPI Service
  │
  ▼
ONNX Runtime Inference Engine
  │
  ▼
MiniLM Sentiment Classifier
  │
  ▼
Prediction (positive / neutral / negative)

Monitoring Layer:
  ├── Prometheus (/metrics)
  └── Logging (structured logs)
```

Key design principle:

> Training uses transformer models, inference uses ONNX Runtime for speed.

---

## Model Choice

The system uses:

* MiniLM (primary model)
* Alternative baseline: DistilBERT

Why:

* Small footprint
* Fast CPU inference
* Strong performance on sentiment classification
* Suitable for ONNX export

---

## Training Approach

Sentence-level classification:

* Input: individual sentence review sentences
* Label mapping from star ratings:

  * 1–2 → negative
  * 3 → neutral
  * 4–5 → positive

Multi-sentence reviews:

* Each sentence inherits review-level rating

We split the dataset with train/test ratio of 80/20, ensuring stratified sampling to maintain label distribution.
After we preprocess the data to split in to individual sentences, we fine-tune MiniLM on the training set and evaluate on the test set.

---

## Inference Stack

* FastAPI for serving requests
* ONNX Runtime for low-latency inference
* MiniLM exported to ONNX format

---

## Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI server
│   ├── model.py             # ONNX inference wrapper
│   ├── metrics.py           # Prometheus metrics
│   └── schema.py            # request/response schemas
│
├── training/
│   ├── train.py            # fine-tuning script
│   └── preprocess.py       # dataset preparation
│
├── onnx_model/
│   └── model.onnx
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

### 1. Create environment

```bash
conda create -n sentiment python=3.10 -y
conda activate sentiment
```

---

### 2. Install dependencies

```bash
pip install fastapi uvicorn transformers optimum[onnxruntime] onnxruntime prometheus-client numpy
```

---

## Running Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running Training

Run the training script from the repository root using the project's `uv` runner:

```bash
uv run python -m training.train
```

This will execute `training/train.py` as a module.

---

## API Usage

### Predict sentiment

```bash
POST /predict
```

Request:

```json
{
  "input_ids": [...],
  "attention_mask": [...]
}
```

Response:

```json
{
  "prediction": "positive",
  "latency_ms": 42.3
}
```

---

## Monitoring

### Prometheus metrics endpoint

```
GET /metrics
```

Exposed metrics:

* request count
* latency histogram
* error rate (optional extension)

---

## Logging

Structured logs capture:

* request latency
* model inference time
* prediction distribution
* error traces

---

## Docker Deployment

```bash
docker build -t sentiment-service .
docker run -p 8000:8000 sentiment-service
```

---

## Cloud Readiness

The system is cloud-compatible without modification:

* Stateless API design
* Dockerized service
* Environment-based configuration
* CPU or GPU inference selectable

Deployable to:

* Kubernetes (EKS / AKS / GKE)
* AWS ECS
* Azure Container Apps
* Google Cloud Run

---

## Performance Targets

On CPU:

* P50 latency: ~20–50 ms
* P95 latency: ~50–120 ms
* P99 latency: < 150–250 ms

Meets requirement:

> P99 < 300 ms

---

## Key Design Decisions

* ONNX Runtime chosen for deterministic low-latency inference
* MiniLM selected for best speed/accuracy trade-off
* FastAPI chosen for lightweight production API layer
* Prometheus used for observability

---

## Possible Improvements

* INT8 quantization for further latency reduction
* Batch inference endpoint
* GPU acceleration via CUDA EP
* Kafka-based async ingestion pipeline
* Model versioning (MLflow)
* **Kubeflow Pipelines** – Introduce after POC validation. Start with local/manual workflows, containerize with Docker, then add Kubeflow for orchestration once the training→inference pipeline is proven and reproducible.

---

## Summary

This system demonstrates a production-ready NLP inference pipeline with:

* Efficient transformer model (MiniLM)
* Optimized inference (ONNX Runtime)
* Observable API layer (FastAPI + Prometheus)
* Cloud-native deployment design
