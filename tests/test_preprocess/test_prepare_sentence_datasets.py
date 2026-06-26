import pytest

from training.preprocess import prepare_sentence_datasets


@pytest.fixture(scope="module")
def data_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "reviews.json"

    path.write_text("""
    [
      {"text": "Great product. Works well.", "rating": 5},
      {"text": "Bad quality. Broke quickly.", "rating": 1}
    ]
    """.strip())

    return str(path)


def test_prepare_sentence_datasets_integration(data_file):
    result = prepare_sentence_datasets(
        data_files=data_file,
        test_size=0.5,
        seed=42,
        label_map={1: 0, 5: 2},
        spacy_model="en_core_web_md",
    )

    train = result["train"]
    test = result["test"]

    # basic structure checks
    assert "text" in train.column_names
    assert "review_rating" in train.column_names

    assert len(train) > 0
    assert len(test) > 0

    # label correctness (no unknown labels)
    assert set(train["review_rating"]).issubset({0, 2})
    assert set(test["review_rating"]).issubset({0, 2})

    # sentence split happened (should be more than 2 rows total)
    assert len(train) + len(test) >= 4

def test_split_stability_deterministic():
    a = prepare_sentence_datasets("tests/data/reviews.json", seed=42)
    b = prepare_sentence_datasets("tests/data/reviews.json", seed=42)

    assert len(a["train"]) == len(b["train"])
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