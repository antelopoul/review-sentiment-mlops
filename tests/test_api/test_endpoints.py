"""
Integration tests for FastAPI inference endpoints.

Covers:
  GET  /health
  POST /predict        — single text
  POST /predict/batch  — up to 64 texts

The ONNX session and tokenizer are stubbed (see conftest.py); all tests
exercise HTTP-level behaviour, schema contracts, and input validation.
"""

import pytest

# Must match the value injected in conftest.py — no relative import needed;
# pytest auto-discovers conftest.py and injects its fixtures directly.
TEST_API_KEY = "test-sentinel-key-abc123"

AUTH = {"X-API-Key": TEST_API_KEY}
WRONG_AUTH = {"X-API-Key": "not-the-right-key"}
VALID_LABELS = {"negative", "neutral", "positive"}

VALID_TEXT = "This product is absolutely amazing!"
BOUNDARY_TEXT = "A" * 1000   # exactly at the 1000-char limit — must be accepted
OVER_LIMIT_TEXT = "A" * 1001  # one char over — must be rejected


# ---------------------------------------------------------------------------
# Response-shape helpers
# ---------------------------------------------------------------------------

def _assert_single_response(data: dict) -> None:
    """Assert that a dict has the shape of SentimentResponseV1."""
    assert data["schema_version"] == "sentiment_response_v1"
    assert data["prediction"] in VALID_LABELS
    assert isinstance(data["label"], int)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["logits"], list)
    assert all(isinstance(v, float) for v in data["logits"])
    assert isinstance(data["model_version"], str)
    assert isinstance(data["request_id"], str)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_is_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_no_auth_required(self, client):
        """Health endpoint must be reachable without an API key."""
        r = client.get("/health")  # no AUTH header
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:

    # --- Happy path ---

    def test_valid_request_returns_200(self, client):
        r = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        assert r.status_code == 200

    def test_response_schema(self, client):
        r = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        _assert_single_response(r.json())

    def test_prediction_is_positive(self, client):
        """Stub logits [1.0, 0.5, 3.0] → argmax=2 → 'positive'."""
        r = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        assert r.json()["prediction"] == "positive"
        assert r.json()["label"] == 2

    def test_confidence_is_between_0_and_1(self, client):
        r = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        assert 0.0 < r.json()["confidence"] < 1.0

    def test_logits_has_three_values(self, client):
        """Three-class model must return exactly 3 logit values."""
        r = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        assert len(r.json()["logits"]) == 3

    def test_request_id_is_unique_per_call(self, client):
        r1 = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        r2 = client.post("/predict", json={"text": VALID_TEXT}, headers=AUTH)
        assert r1.json()["request_id"] != r2.json()["request_id"]

    def test_boundary_length_text_accepted(self, client):
        """Text of exactly MAX_INPUT_CHARS (1000) must not be rejected."""
        r = client.post("/predict", json={"text": BOUNDARY_TEXT}, headers=AUTH)
        assert r.status_code == 200

    # --- Auth ---

    def test_missing_api_key_returns_403(self, client):
        r = client.post("/predict", json={"text": VALID_TEXT})  # no header
        assert r.status_code == 403

    def test_wrong_api_key_returns_403(self, client):
        r = client.post("/predict", json={"text": VALID_TEXT}, headers=WRONG_AUTH)
        assert r.status_code == 403

    # --- Input validation ---

    def test_empty_text_returns_422(self, client):
        r = client.post("/predict", json={"text": ""}, headers=AUTH)
        assert r.status_code == 422

    def test_whitespace_only_text_returns_422(self, client):
        r = client.post("/predict", json={"text": "   "}, headers=AUTH)
        assert r.status_code == 422

    def test_text_over_limit_returns_422(self, client):
        r = client.post("/predict", json={"text": OVER_LIMIT_TEXT}, headers=AUTH)
        assert r.status_code == 422

    def test_missing_text_field_returns_422(self, client):
        r = client.post("/predict", json={}, headers=AUTH)
        assert r.status_code == 422

    def test_schema_version_field_is_optional_in_request(self, client):
        """schema_version has a default; omitting it must not cause a 422."""
        r = client.post(
            "/predict",
            json={"schema_version": "sentiment_request_v1", "text": VALID_TEXT},
            headers=AUTH,
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:

    # --- Happy path ---

    def test_valid_batch_returns_200(self, client):
        r = client.post(
            "/predict/batch",
            json={"texts": ["Great product!", "It was okay.", "Terrible."]},
            headers=AUTH,
        )
        assert r.status_code == 200

    def test_response_schema(self, client):
        texts = ["Great!", "Okay.", "Bad."]
        r = client.post("/predict/batch", json={"texts": texts}, headers=AUTH)
        data = r.json()
        assert data["schema_version"] == "sentiment_batch_response_v1"
        assert isinstance(data["model_version"], str)
        assert isinstance(data["request_id"], str)
        assert "results" in data

    def test_result_count_matches_input(self, client):
        texts = ["One.", "Two.", "Three.", "Four."]
        r = client.post("/predict/batch", json={"texts": texts}, headers=AUTH)
        assert len(r.json()["results"]) == len(texts)

    def test_each_result_has_valid_shape(self, client):
        texts = ["Good product.", "Bad product."]
        r = client.post("/predict/batch", json={"texts": texts}, headers=AUTH)
        for result in r.json()["results"]:
            _assert_single_response(result)

    def test_single_item_batch(self, client):
        r = client.post("/predict/batch", json={"texts": [VALID_TEXT]}, headers=AUTH)
        assert r.status_code == 200
        assert len(r.json()["results"]) == 1

    def test_max_batch_size_accepted(self, client):
        """64 texts is the declared limit and must be accepted."""
        texts = [VALID_TEXT] * 64
        r = client.post("/predict/batch", json={"texts": texts}, headers=AUTH)
        assert r.status_code == 200

    def test_boundary_length_text_in_batch_accepted(self, client):
        r = client.post(
            "/predict/batch",
            json={"texts": [BOUNDARY_TEXT, VALID_TEXT]},
            headers=AUTH,
        )
        assert r.status_code == 200

    # --- Auth ---

    def test_missing_api_key_returns_403(self, client):
        r = client.post("/predict/batch", json={"texts": [VALID_TEXT]})
        assert r.status_code == 403

    def test_wrong_api_key_returns_403(self, client):
        r = client.post("/predict/batch", json={"texts": [VALID_TEXT]}, headers=WRONG_AUTH)
        assert r.status_code == 403

    # --- Input validation ---

    def test_empty_texts_list_returns_422(self, client):
        r = client.post("/predict/batch", json={"texts": []}, headers=AUTH)
        assert r.status_code == 422

    def test_batch_over_limit_returns_422(self, client):
        """65 texts exceeds MAX_BATCH_SIZE (64) and must be rejected."""
        texts = [VALID_TEXT] * 65
        r = client.post("/predict/batch", json={"texts": texts}, headers=AUTH)
        assert r.status_code == 422

    def test_empty_string_in_batch_returns_422(self, client):
        r = client.post(
            "/predict/batch",
            json={"texts": [VALID_TEXT, ""]},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_whitespace_only_in_batch_returns_422(self, client):
        r = client.post(
            "/predict/batch",
            json={"texts": [VALID_TEXT, "   "]},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_over_limit_text_in_batch_returns_422(self, client):
        r = client.post(
            "/predict/batch",
            json={"texts": [VALID_TEXT, OVER_LIMIT_TEXT]},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_missing_texts_field_returns_422(self, client):
        r = client.post("/predict/batch", json={}, headers=AUTH)
        assert r.status_code == 422
