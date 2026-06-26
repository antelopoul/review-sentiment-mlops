from training.preprocess import review_to_sentence_samples


def test_sentence_segmentation_title_and_text(nlp):
    review = {
        "rating": 5,
        "title": "Great",
        "text": "Works well. Highly recommended."
    }

    label_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

    result = review_to_sentence_samples(review, label_map=label_map, nlp=nlp)

    texts = [r["text"] for r in result]
    labels = [r["labels"] for r in result]
    review_ratings = [r["review_rating"] for r in result]

    assert "Works well." in texts
    assert "Highly recommended." in texts
    assert all(label == 2 for label in labels)
    assert all(rating == 5 for rating in review_ratings)
