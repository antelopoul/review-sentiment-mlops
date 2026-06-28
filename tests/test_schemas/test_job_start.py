"""
Unit tests for all Pydantic schemas defined in app/schema.py.

Coverage:
  - Valid construction and field defaults
  - Required-field enforcement (ValidationError when missing)
  - Literal / enum constraint enforcement
  - Numeric bounds (progress ge=0, le=1)
  - model_dump() key contract
  - Round-trip: model_validate(model.model_dump())
  - Nested model composition
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from app.schema import (
    JobInput,
    JobMetadata,
    SentimentRequestV1,
    SentimentResponseV1,
    SentimentBatchRequestV1,
    SentimentBatchResponseV1,
    SentimentJobStartV1,
    SentimentJobStatusV1,
    SentimentResult,
    SentimentMetrics,
    SentimentModelInfo,
    SentimentError,
    SentimentJobResultV1,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

def _response() -> SentimentResponseV1:
    """Minimal valid SentimentResponseV1 for use in batch tests."""
    return SentimentResponseV1(
        prediction="positive",
        label=2,
        confidence=0.95,
        logits=[-1.0, 0.5, 3.2],
        model_version="1.0.0",
        request_id="abc123",
    )


# ---------------------------------------------------------------------------
# JobInput
# ---------------------------------------------------------------------------

class TestJobInput:
    def test_valid(self):
        obj = JobInput(text="some review text")
        assert obj.text == "some review text"

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            JobInput()

    def test_contract(self):
        assert set(JobInput(text="x").model_dump().keys()) == {"text"}


# ---------------------------------------------------------------------------
# JobMetadata
# ---------------------------------------------------------------------------

class TestJobMetadata:
    def test_all_optional_defaults_to_none(self):
        obj = JobMetadata()
        assert obj.helpful_vote is None
        assert obj.verified_purchase is None

    def test_with_values(self):
        obj = JobMetadata(helpful_vote="yes", verified_purchase="true")
        assert obj.helpful_vote == "yes"
        assert obj.verified_purchase == "true"

    def test_contract(self):
        keys = set(JobMetadata().model_dump().keys())
        assert keys == {"helpful_vote", "verified_purchase"}


# ---------------------------------------------------------------------------
# SentimentRequestV1
# ---------------------------------------------------------------------------

class TestSentimentRequestV1:
    def test_valid(self):
        obj = SentimentRequestV1(text="Great product!")
        assert obj.text == "Great product!"
        assert obj.schema_version == "sentiment_request_v1"

    def test_schema_version_default(self):
        obj = SentimentRequestV1(text="x")
        assert obj.schema_version == "sentiment_request_v1"

    def test_wrong_schema_version_raises(self):
        with pytest.raises(ValidationError):
            SentimentRequestV1(schema_version="wrong_version", text="x")

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            SentimentRequestV1()

    def test_contract(self):
        keys = set(SentimentRequestV1(text="x").model_dump().keys())
        assert keys == {"schema_version", "text"}


# ---------------------------------------------------------------------------
# SentimentResponseV1
# ---------------------------------------------------------------------------

class TestSentimentResponseV1:
    def test_valid_positive(self):
        obj = _response()
        assert obj.prediction == "positive"
        assert obj.label == 2
        assert obj.schema_version == "sentiment_response_v1"

    @pytest.mark.parametrize("prediction", ["negative", "neutral", "positive"])
    def test_all_valid_predictions(self, prediction):
        obj = SentimentResponseV1(
            prediction=prediction,
            label=0,
            confidence=0.9,
            logits=[1.0, 2.0, 3.0],
            model_version="1.0.0",
            request_id="id1",
        )
        assert obj.prediction == prediction

    def test_invalid_prediction_raises(self):
        with pytest.raises(ValidationError):
            SentimentResponseV1(
                prediction="unknown",
                label=0,
                confidence=0.9,
                logits=[1.0],
                model_version="1.0.0",
                request_id="id1",
            )

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SentimentResponseV1(
                prediction="positive",
                # label missing
                confidence=0.9,
                logits=[1.0],
                model_version="1.0.0",
                request_id="id1",
            )

    def test_contract(self):
        keys = set(_response().model_dump().keys())
        assert keys == {"schema_version", "prediction", "label", "confidence", "logits", "model_version", "request_id"}

    def test_roundtrip(self):
        obj = _response()
        assert SentimentResponseV1.model_validate(obj.model_dump()) == obj


# ---------------------------------------------------------------------------
# SentimentBatchRequestV1
# ---------------------------------------------------------------------------

class TestSentimentBatchRequestV1:
    def test_valid(self):
        obj = SentimentBatchRequestV1(texts=["hello", "world"])
        assert len(obj.texts) == 2
        assert obj.schema_version == "sentiment_batch_request_v1"

    def test_missing_texts_raises(self):
        with pytest.raises(ValidationError):
            SentimentBatchRequestV1()

    def test_contract(self):
        keys = set(SentimentBatchRequestV1(texts=["x"]).model_dump().keys())
        assert keys == {"schema_version", "texts"}


# ---------------------------------------------------------------------------
# SentimentBatchResponseV1
# ---------------------------------------------------------------------------

class TestSentimentBatchResponseV1:
    def test_valid(self):
        obj = SentimentBatchResponseV1(
            results=[_response(), _response()],
            model_version="1.0.0",
            request_id="batch-abc",
        )
        assert len(obj.results) == 2
        assert obj.schema_version == "sentiment_batch_response_v1"

    def test_empty_results_allowed(self):
        obj = SentimentBatchResponseV1(results=[], model_version="1.0.0", request_id="r")
        assert obj.results == []

    def test_nested_result_type_enforced(self):
        with pytest.raises((ValidationError, TypeError)):
            SentimentBatchResponseV1(
                results=["not-a-response-object"],
                model_version="1.0.0",
                request_id="r",
            )

    def test_contract(self):
        keys = set(
            SentimentBatchResponseV1(results=[], model_version="1.0.0", request_id="r")
            .model_dump()
            .keys()
        )
        assert keys == {"schema_version", "results", "model_version", "request_id"}


# ---------------------------------------------------------------------------
# SentimentJobStartV1
# ---------------------------------------------------------------------------

class TestSentimentJobStartV1:
    def _make(self, **kwargs):
        defaults = dict(
            job_id="job-001",
            event="job_start",
            timestamp=NOW,
            input=JobInput(text="review text"),
            metadata=JobMetadata(),
        )
        defaults.update(kwargs)
        return SentimentJobStartV1(**defaults)

    def test_valid(self):
        obj = self._make()
        assert obj.job_id == "job-001"
        assert obj.status == "queued"  # default
        assert obj.schema_version == "sentiment_job_start_v1"

    def test_status_default_is_queued(self):
        obj = self._make()
        assert obj.status == "queued"

    @pytest.mark.parametrize("status", ["queued", "running", "success", "failed", "timeout"])
    def test_all_valid_statuses(self, status):
        obj = self._make(status=status)
        assert obj.status == status

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            self._make(status="pending")

    def test_invalid_event_raises(self):
        with pytest.raises(ValidationError):
            self._make(event="job_done")

    def test_missing_job_id_raises(self):
        with pytest.raises(ValidationError):
            SentimentJobStartV1(
                event="job_start",
                timestamp=NOW,
                input=JobInput(text="x"),
                metadata=JobMetadata(),
            )

    def test_nested_input_accepted(self):
        obj = self._make(input=JobInput(text="nested text"))
        assert obj.input.text == "nested text"

    def test_contract(self):
        keys = set(self._make().model_dump().keys())
        assert keys == {"schema_version", "job_id", "event", "timestamp", "input", "metadata", "status"}


# ---------------------------------------------------------------------------
# SentimentJobStatusV1
# ---------------------------------------------------------------------------

class TestSentimentJobStatusV1:
    def _make(self, **kwargs):
        defaults = dict(
            job_id="job-001",
            event="status_update",
            timestamp=NOW,
            status="running",
            progress=0.5,
        )
        defaults.update(kwargs)
        return SentimentJobStatusV1(**defaults)

    def test_valid(self):
        obj = self._make()
        assert obj.progress == 0.5
        assert obj.message is None

    def test_message_optional(self):
        obj = self._make(message="halfway done")
        assert obj.message == "halfway done"

    def test_progress_boundary_values(self):
        assert self._make(progress=0.0).progress == 0.0
        assert self._make(progress=1.0).progress == 1.0

    def test_progress_below_zero_raises(self):
        with pytest.raises(ValidationError):
            self._make(progress=-0.01)

    def test_progress_above_one_raises(self):
        with pytest.raises(ValidationError):
            self._make(progress=1.01)

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            self._make(status="unknown")

    def test_invalid_event_raises(self):
        with pytest.raises(ValidationError):
            self._make(event="job_start")

    def test_contract(self):
        keys = set(self._make().model_dump().keys())
        assert keys == {"schema_version", "job_id", "event", "timestamp", "status", "progress", "message"}


# ---------------------------------------------------------------------------
# SentimentResult
# ---------------------------------------------------------------------------

class TestSentimentResult:
    @pytest.mark.parametrize("label", ["positive", "neutral", "negative"])
    def test_all_valid_labels(self, label):
        obj = SentimentResult(label=label, confidence=0.8)
        assert obj.label == label

    def test_invalid_label_raises(self):
        with pytest.raises(ValidationError):
            SentimentResult(label="mixed", confidence=0.5)

    def test_missing_confidence_raises(self):
        with pytest.raises(ValidationError):
            SentimentResult(label="positive")

    def test_contract(self):
        keys = set(SentimentResult(label="positive", confidence=0.9).model_dump().keys())
        assert keys == {"label", "confidence"}


# ---------------------------------------------------------------------------
# SentimentMetrics
# ---------------------------------------------------------------------------

class TestSentimentMetrics:
    def test_valid(self):
        obj = SentimentMetrics(latency_ms=12.5, model_latency_ms=10.0, tokenization_ms=2.5)
        assert obj.latency_ms == 12.5

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            SentimentMetrics(latency_ms=10.0, model_latency_ms=8.0)

    def test_contract(self):
        keys = set(
            SentimentMetrics(latency_ms=1.0, model_latency_ms=0.8, tokenization_ms=0.2)
            .model_dump()
            .keys()
        )
        assert keys == {"latency_ms", "model_latency_ms", "tokenization_ms"}


# ---------------------------------------------------------------------------
# SentimentModelInfo
# ---------------------------------------------------------------------------

class TestSentimentModelInfo:
    def test_valid(self):
        obj = SentimentModelInfo(name="all-MiniLM-L6-v2", version="1.0.0")
        assert obj.name == "all-MiniLM-L6-v2"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            SentimentModelInfo(version="1.0.0")

    def test_contract(self):
        keys = set(SentimentModelInfo(name="m", version="v").model_dump().keys())
        assert keys == {"name", "version"}


# ---------------------------------------------------------------------------
# SentimentError
# ---------------------------------------------------------------------------

class TestSentimentError:
    def test_valid(self):
        obj = SentimentError(type="InferenceError", message="ONNX session failed")
        assert obj.type == "InferenceError"

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            SentimentError(message="oops")

    def test_contract(self):
        keys = set(SentimentError(type="E", message="m").model_dump().keys())
        assert keys == {"type", "message"}


# ---------------------------------------------------------------------------
# SentimentJobResultV1
# ---------------------------------------------------------------------------

class TestSentimentJobResultV1:
    def _make(self, **kwargs):
        defaults = dict(
            job_id="job-001",
            event="job_result",
            timestamp=NOW,
            status="success",
        )
        defaults.update(kwargs)
        return SentimentJobResultV1(**defaults)

    def test_valid_minimal(self):
        obj = self._make()
        assert obj.result is None
        assert obj.metrics is None
        assert obj.model is None
        assert obj.error is None
        assert obj.schema_version == "sentiment_job_result_v1"

    def test_valid_with_all_optionals(self):
        obj = self._make(
            result=SentimentResult(label="positive", confidence=0.95),
            metrics=SentimentMetrics(latency_ms=12.0, model_latency_ms=10.0, tokenization_ms=2.0),
            model=SentimentModelInfo(name="MiniLM", version="1.0.0"),
        )
        assert obj.result.label == "positive"
        assert obj.metrics.latency_ms == 12.0
        assert obj.model.name == "MiniLM"

    def test_failed_job_with_error(self):
        obj = self._make(
            status="failed",
            error=SentimentError(type="TimeoutError", message="Inference timed out"),
        )
        assert obj.status == "failed"
        assert obj.error.type == "TimeoutError"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            self._make(status="cancelled")

    def test_invalid_event_raises(self):
        with pytest.raises(ValidationError):
            self._make(event="job_start")

    def test_contract(self):
        keys = set(self._make().model_dump().keys())
        assert keys == {
            "schema_version", "job_id", "event", "timestamp",
            "status", "result", "metrics", "model", "error",
        }

    def test_roundtrip(self):
        obj = self._make(
            status="success",
            result=SentimentResult(label="neutral", confidence=0.72),
        )
        assert SentimentJobResultV1.model_validate(obj.model_dump()) == obj
