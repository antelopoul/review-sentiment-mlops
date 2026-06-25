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