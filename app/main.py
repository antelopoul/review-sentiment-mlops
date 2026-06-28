"""
FastAPI ONNX Runtime inference service for sentiment analysis.
Loads model and tokenizer once at startup; performs warmup inference.
Batch inference is used for all predictions to maximise throughput and minimise per-item latency.
"""

import collections
import os
import time
import logging
import secrets
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import Response
from transformers import AutoTokenizer

from app.schema import (
    SentimentRequestV1,
    SentimentResponseV1,
    SentimentBatchRequestV1,
    SentimentBatchResponseV1,
)
from app.metrics import (
    metrics_response,
    request_latency_seconds,
    requests_total,
    predictions_total,
    confidence_histogram,
    confidence_by_class,
    errors_total,
    invalid_input_total,
    p95_latency_seconds,
    p99_latency_seconds,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "./onnx_model/model.int8.onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./models/sentimentv1")
WARMUP_TEXT = "This product is really good and works well."
MAX_INPUT_CHARS = 1000
MAX_BATCH_SIZE = 64
MODEL_VERSION = os.getenv("APP_VERSION", "dev")  # set by Dockerfile ARG BUILD_VERSION
LABEL_MAP: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

_latency_window: collections.deque[float] = collections.deque(maxlen=1000)

API_KEY = os.environ.get("API_KEY", "")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(_api_key_header)) -> None:
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals — populated once in lifespan, never re-initialised per request
# ---------------------------------------------------------------------------

session: ort.InferenceSession = None
tokenizer: AutoTokenizer = None
model_input_names: set[str] = set()

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, tokenizer, model_input_names, API_KEY

    if not API_KEY:
        API_KEY = secrets.token_urlsafe(32)
        logger.warning("No API_KEY env var set — generated one for this session:")
        logger.warning("  X-API-Key: %s", API_KEY)
        logger.warning("Set API_KEY in your environment or .env file to make it permanent.")

    logger.info("Loading tokenizer from %s ...", TOKENIZER_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    logger.info("Loading ONNX model from %s ...", MODEL_PATH)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    model_input_names = {inp.name for inp in session.get_inputs()}
    logger.info("Model input names: %s", model_input_names)

    logger.info("Running warmup inference ...")
    t0 = time.perf_counter()
    _run_inference([WARMUP_TEXT])
    logger.info("Warmup done in %.1f ms", (time.perf_counter() - t0) * 1000)

    yield

    logger.info("Shutting down — releasing ONNX session.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentiment Classification",
    description="INT8 ONNX Runtime inference endpoint for sentiment classification.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Core inference — always batched: single ONNX call for N texts
# ---------------------------------------------------------------------------


def _run_inference(texts: list[str]) -> list[tuple[list[float], int]]:
    """Tokenize batch, run one ONNX session call, return (logits, predicted_class) per item."""
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")

    inputs: dict[str, np.ndarray] = {}
    if "input_ids" in model_input_names:
        inputs["input_ids"] = encoded["input_ids"].astype(np.int64)
    if "attention_mask" in model_input_names:
        inputs["attention_mask"] = encoded["attention_mask"].astype(np.int64)
    if "token_type_ids" in model_input_names:
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(encoded["input_ids"], dtype=np.int64)
        inputs["token_type_ids"] = token_type_ids.astype(np.int64)

    logits_batch: np.ndarray = session.run(None, inputs)[0]  # (batch, num_classes)
    return [(row.tolist(), int(np.argmax(row))) for row in logits_batch]


def _to_response(logits: list[float], predicted_class: int) -> SentimentResponseV1:
    """Build response and observe per-prediction metrics."""
    arr = np.array(logits)
    exp_l = np.exp(arr - arr.max())
    confidence = float(exp_l[predicted_class] / exp_l.sum())
    label = LABEL_MAP[predicted_class]

    predictions_total.labels(label=label).inc()
    confidence_histogram.observe(confidence)
    confidence_by_class.labels(predicted_class=label).observe(confidence)

    return SentimentResponseV1(
        prediction=label,
        label=predicted_class,
        confidence=confidence,
        logits=logits,
        model_version=MODEL_VERSION,
        request_id=secrets.token_hex(8),
    )


def _observe_latency(elapsed_s: float) -> None:
    request_latency_seconds.observe(elapsed_s)
    _latency_window.append(elapsed_s)
    if len(_latency_window) >= 2:
        arr = np.array(_latency_window)
        p95_latency_seconds.set(float(np.percentile(arr, 95)))
        p99_latency_seconds.set(float(np.percentile(arr, 99)))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=SentimentResponseV1, dependencies=[Security(verify_api_key)])
def predict(request: SentimentRequestV1) -> SentimentResponseV1:
    if not request.text or not request.text.strip():
        invalid_input_total.inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=422, detail="'text' must be a non-empty string.")

    if len(request.text) > MAX_INPUT_CHARS:
        invalid_input_total.inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(
            status_code=422,
            detail=f"'text' exceeds {MAX_INPUT_CHARS} characters (got {len(request.text)}).",
        )

    t0 = time.perf_counter()
    try:
        [(logits, predicted_class)] = _run_inference([request.text])
    except Exception as exc:
        errors_total.labels(error_type=type(exc).__name__).inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    elapsed_s = time.perf_counter() - t0
    result = _to_response(logits, predicted_class)
    requests_total.labels(model=MODEL_VERSION, status="success").inc()
    _observe_latency(elapsed_s)

    logger.info("predict | class=%s | conf=%.3f | latency=%.2f ms", result.prediction, result.confidence, elapsed_s * 1000)
    return result


@app.post("/predict/batch", response_model=SentimentBatchResponseV1, dependencies=[Security(verify_api_key)])
def predict_batch(request: SentimentBatchRequestV1) -> SentimentBatchResponseV1:
    if not request.texts:
        invalid_input_total.inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=422, detail="'texts' must be a non-empty list.")

    if len(request.texts) > MAX_BATCH_SIZE:
        invalid_input_total.inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=422, detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE}.")

    empty = [i for i, t in enumerate(request.texts) if not t or not t.strip()]
    if empty:
        invalid_input_total.inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=422, detail=f"Empty texts at indices: {empty}.")

    oversized = [i for i, t in enumerate(request.texts) if len(t) > MAX_INPUT_CHARS]
    if oversized:
        invalid_input_total.inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=422, detail=f"Texts exceed {MAX_INPUT_CHARS} chars at indices: {oversized}.")

    t0 = time.perf_counter()
    try:
        batch_results = _run_inference(request.texts)
    except Exception as exc:
        errors_total.labels(error_type=type(exc).__name__).inc()
        requests_total.labels(model=MODEL_VERSION, status="error").inc()
        raise HTTPException(status_code=500, detail="Inference failed.") from exc

    elapsed_s = time.perf_counter() - t0
    results = [_to_response(logits, cls) for logits, cls in batch_results]
    requests_total.labels(model=MODEL_VERSION, status="success").inc()
    _observe_latency(elapsed_s / len(request.texts))  # per-item latency keeps p95/p99 comparable

    logger.info(
        "predict/batch | size=%d | total=%.2f ms | per-item=%.2f ms",
        len(request.texts), elapsed_s * 1000, elapsed_s * 1000 / len(request.texts),
    )
    return SentimentBatchResponseV1(
        results=results,
        model_version=MODEL_VERSION,
        request_id=secrets.token_hex(8),
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": MODEL_VERSION,
        "model": MODEL_PATH,
        "tokenizer": TOKENIZER_PATH,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(metrics_response(), media_type="text/plain; version=0.0.4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
