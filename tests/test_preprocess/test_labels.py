from training.preprocess import review_to_sentence_samples


def test_label_mapping(nlp):
    label_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

    review = {"rating": 1, "text": "Bad product"}

    result = review_to_sentence_samples(review, label_map=label_map, nlp=nlp)

    assert result[0]["labels"] == 0
    assert result[0]["review_rating"] == 1


def test_default_label_map_used(nlp):
    review = {"rating": 5, "text": "Good"}

    result = review_to_sentence_samples(review, nlp=nlp)

    assert result[0]["labels"] == 2
    assert result[0]["review_rating"] == 5

def test_label_invariant(nlp):
    review = {"rating": 5, "text": "Good. Excellent."}

    label_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

    result = review_to_sentence_samples(review, label_map=label_map, nlp=nlp)

    labels = {r["labels"] for r in result}

    assert len(labels) == 1  # all sentences inherit same label
