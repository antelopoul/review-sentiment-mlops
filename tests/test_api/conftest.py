"""
Shared fixtures for API endpoint tests.

Spins up the FastAPI app with the ONNX session and HuggingFace tokenizer
replaced by lightweight stubs so no real model files are needed.

Stub behaviour:
  - Tokenizer returns real numpy arrays (shape: (n, 10)) so `.astype()` and
    `len()` work correctly through the inference path.
  - ONNX session always returns logits [1.0, 0.5, 3.0], which predict class 2
    ("positive") for every input.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

TEST_API_KEY = "test-sentinel-key-abc123"

# Logits that always resolve to class 2 ("positive") via argmax
_STUB_LOGITS = [1.0, 0.5, 3.0]


def _make_tokenizer_stub() -> MagicMock:
    """
    Callable mock that mimics AutoTokenizer behaviour.
    Returns a plain dict with numpy arrays so downstream code can call
    .astype() and .get() without issues.
    """
    stub = MagicMock()

    def _tokenize(texts, **kwargs):
        n = len(texts)
        return {
            "input_ids": np.ones((n, 10), dtype=np.int64),
            "attention_mask": np.ones((n, 10), dtype=np.int64),
            # token_type_ids intentionally absent — exercises the fallback path
        }

    stub.side_effect = _tokenize
    return stub


def _make_session_stub() -> MagicMock:
    """
    Mock ort.InferenceSession.

    get_inputs() returns objects for 'input_ids' and 'attention_mask' so
    model_input_names is populated correctly during lifespan startup.
    run() returns one logit row per item in the batch.
    """
    stub = MagicMock()

    inp_ids = MagicMock()
    inp_ids.name = "input_ids"
    attn = MagicMock()
    attn.name = "attention_mask"
    stub.get_inputs.return_value = [inp_ids, attn]

    stub.run.side_effect = lambda _, inputs: [
        np.array([_STUB_LOGITS] * len(inputs["input_ids"]), dtype=np.float32)
    ]
    return stub


@pytest.fixture(scope="module")
def client() -> TestClient:
    """
    Module-scoped TestClient.  The lifespan runs once per module — model load
    and warmup inference both use stubs, completing in microseconds.

    API_KEY is injected into the module global before the lifespan starts so
    the server does not regenerate it; tests can then use TEST_API_KEY.
    """
    with (
        patch("app.main.ort.InferenceSession", return_value=_make_session_stub()),
        patch("app.main.AutoTokenizer.from_pretrained", return_value=_make_tokenizer_stub()),
    ):
        import app.main as main_module

        # Set before lifespan runs: the `if not API_KEY` branch is skipped,
        # so the key stays fixed for the duration of this test module.
        main_module.API_KEY = TEST_API_KEY

        from app.main import app
        with TestClient(app) as c:
            yield c
