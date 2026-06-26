import pytest
from pathlib import Path

from training.preprocess import prepare_sentence_datasets
import os

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_DATASETS_CACHE"] = "C:/hf_cache"

@pytest.fixture(scope="module")
def data_file():
    # Find the file relative to this test file location (goes up to project root)
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "tests" / "data" / "reviews.json"
    
    return str(path)


def test_prepare_sentence_datasets_integration(data_file):
    result = prepare_sentence_datasets(
        data_files=data_file,
        val_size=0.1,
        test_size=0.1,
        seed=42,
        label_map = {
            1: 0,  # negative
            2: 0,  # negative
            3: 1,  # neutral
            4: 2,  # positive
            5: 2,  # positive
        },
        spacy_model="en_core_web_md",
        disable_components=["ner", "tagger"],
        min_sentence_length=2
    )

    train = result["train"]
    val = result["val"]
    test = result["test"]

    # basic structure checks
    assert "text" in train.column_names
    assert "labels" in train.column_names

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    # label correctness (no unknown labels)
    assert set(train["labels"]).issubset({0, 1, 2})
    assert set(val["labels"]).issubset({0, 1, 2})
    assert set(test["labels"]).issubset({0, 1, 2})

    # sentence split happened (should be more than 2 rows total)
    assert len(train) + len(val) + len(test) >= 4

def test_split_stability_deterministic():
    a = prepare_sentence_datasets("tests/data/reviews.json", seed=42)
    b = prepare_sentence_datasets("tests/data/reviews.json", seed=42)

    assert len(a["train"]) == len(b["train"])
    assert len(a["val"]) == len(b["val"])
    assert len(a["test"]) == len(b["test"])

    assert set(a["train"]["text"]) == set(b["train"]["text"])

import pytest
from training.preprocess import prepare_sentence_datasets


def test_no_sentence_level_leakage():
    data_path = "tests/fixtures/sample_reviews.json"

    splits = prepare_sentence_datasets(
        data_files=data_path,
        test_size=0.2,
        seed=42,
    )

    train = splits["train"].to_pandas()
    test = splits["test"].to_pandas()

    # 1. HARD CHECK: identical sentence leakage
    train_texts = set(train["text"].str.strip().tolist())
    test_texts = set(test["text"].str.strip().tolist())

    overlap = train_texts.intersection(test_texts)

    assert len(overlap) == 0, f"Sentence leakage detected: {list(overlap)[:10]}"


def test_no_review_overlap_if_id_exists():
    data_path = "tests/data/reviews.json"

    splits = prepare_sentence_datasets(
        data_files=data_path,
        test_size=0.2,
        seed=42,
    )

    train = splits["train"].to_pandas()
    test = splits["test"].to_pandas()

    if "review_id" in train.columns:
        train_ids = set(train["review_id"])
        test_ids = set(test["review_id"])

        overlap = train_ids.intersection(test_ids)

        assert len(overlap) == 0, "Review-level leakage detected"