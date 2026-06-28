from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

registry = CollectorRegistry()

# Latency
request_latency_seconds = Histogram(
    "sentiment_request_latency_seconds",
    "End-to-end request latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
    registry=registry,
)

p95_latency_seconds = Gauge(
    "sentiment_p95_latency_seconds",
    "Rolling p95 end-to-end request latency (last 1000 requests)",
    registry=registry,
)

p99_latency_seconds = Gauge(
    "sentiment_p99_latency_seconds",
    "Rolling p99 end-to-end request latency (last 1000 requests)",
    registry=registry,
)

# Throughput / errors
requests_total = Counter(
    "sentiment_requests_total",
    "Total requests",
    ["model", "status"],
    registry=registry,
)

errors_total = Counter(
    "sentiment_errors_total",
    "Total errors by exception type",
    ["error_type"],
    registry=registry,
)

invalid_input_total = Counter(
    "sentiment_invalid_input_total",
    "Requests rejected due to invalid input",
    registry=registry,
)

# Model output
predictions_total = Counter(
    "sentiment_predictions_total",
    "Prediction counts by label",
    ["label"],
    registry=registry,
)

confidence_histogram = Histogram(
    "sentiment_prediction_confidence",
    "Prediction confidence distribution",
    buckets=(0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
    registry=registry,
)

confidence_by_class = Histogram(
    "sentiment_confidence_by_class",
    "Prediction confidence by predicted class",
    labelnames=["predicted_class"],
    buckets=(0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99),
    registry=registry,
)

# Export
def metrics_response() -> bytes:
    return generate_latest(registry)
