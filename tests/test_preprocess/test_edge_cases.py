from training.preprocess import review_to_sentence_samples


def test_missing_title(nlp):
    review = {"rating": 4, "text": "Works fine."}

    result = review_to_sentence_samples(review, nlp=nlp)

    assert len(result) == 1
    assert result[0]["text"] == "Works fine."


def test_missing_text(nlp):
    review = {"rating": 4, "title": "OK"}

    result = review_to_sentence_samples(review, nlp=nlp)

    assert len(result) == 1
    assert result[0]["text"] == "OK"


def test_empty_review(nlp):
    review = {"rating": 3}

    result = review_to_sentence_samples(review, nlp=nlp)

    assert isinstance(result, list)


def test_short_sentence_filter(nlp):
    review = {
        "rating": 3,
        "text": ". a .. good product."
    }

    result = review_to_sentence_samples(review, nlp=nlp)

    assert all(len(s["text"]) >= 2 for s in result)