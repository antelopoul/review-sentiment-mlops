import pytest
import spacy


@pytest.fixture(scope="session")
def nlp():
    """
    Shared deterministic NLP pipeline for all preprocessing tests.
    """
    nlp = spacy.blank("en")

    # deterministic sentence segmentation
    nlp.add_pipe("sentencizer")

    return nlp