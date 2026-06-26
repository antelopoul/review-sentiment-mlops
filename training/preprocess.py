from re import split
from typing import Optional
import spacy
from datasets import load_dataset, Dataset

LABEL_MAP = {
    1: 0,  # negative
    2: 0,  # negative
    3: 1,  # neutral
    4: 2,  # positive
    5: 2,  # positive
}

def review_to_sentence_samples(review: dict, label_map: dict = None, nlp=None) -> list[dict]:
    """
    Convert a single review into multiple sentence-level samples.
    :review: A dict representing a single review, expected to have keys like "text", "rating", "title", etc.
    :label_map: Mapping from rating to label (e.g., {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}).
    :nlp: spaCy language model for sentence segmentation.
    :return: A list of dicts, each representing a sentence-level sample with associated metadata

    """
    samples = []

    # Use defaults if not provided
    if label_map is None:
        label_map = LABEL_MAP
    if nlp is None:
        nlp = spacy.load("en_core_web_md", disable=["ner", "tagger"])

    # Map the review rating to a label 0 (negative), 1 (neutral), or 2 (positive)
    rating = int(review["rating"])

    # If the rating is not in the label map, return an empty list (no samples)
    if rating not in label_map:
        return []
    
    label = label_map[rating]

    # Extract title and text, handling missing fields gracefully
    title = (review.get("title") or "").strip()
    text = (review.get("text") or "").strip()

    # Use spaCy to split the review into sentences 
    if title:
        for sent in nlp(title).sents:
            s = sent.text.strip()
            # Filter out sentences that are too short or consist only of punctuation
            if len(s.strip()) >= 3:
                samples.append({"text": s, "review_rating": label})
    # Use spaCy to split the review into sentences
    if text:
        for sent in nlp(text).sents:
            s = sent.text.strip()
            # Filter out sentences that are too short or consist only of punctuation
            if len(s.strip()) >= 3:
                samples.append({"text": s, "review_rating": label})

    return samples


def prepare_sentence_datasets(
    data_files: str, 
    test_size: float = 0.1, 
    seed: int = 42,
    label_map: dict = None,
    spacy_model: str = "en_core_web_md",
):
    """
    Load a dataset of reviews, split into train/test, and convert to sentence-level samples.
    :data_files: Path to the data file or archive (supports ZIP archives).
    :test_size: Fraction of the dataset to use for testing.
    :seed: Random seed for reproducibility.
    :label_map: Mapping from rating to label (e.g., {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}).
    :spacy_model: Name of the spaCy model to load.
    :return: A dict with 'train' and 'test' keys containing sentence-level datasets.
    """
    # Load the dataset from the specified files
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    # Load spaCy model
    nlp = spacy.load(spacy_model, disable=["ner", "tagger"])
    
    # Use default label map if not provided
    if label_map is None:
        label_map = LABEL_MAP
    
    # Split the dataset into training and testing sets
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    
    # Process training reviews into sentence-level samples
    train_reviews = split["train"]
    train_sentences = []
    for review in train_reviews:
        train_sentences.extend(review_to_sentence_samples(review, label_map=label_map, nlp=nlp))
    
    # Process testing reviews into sentence-level samples
    test_reviews = split["test"]
    test_sentences = []
    for review in test_reviews:
        test_sentences.extend(review_to_sentence_samples(review, label_map=label_map, nlp=nlp))
    
    # Convert lists of sentence samples back into Hugging Face Datasets
    train_dataset = Dataset.from_dict({
        "text": [sample["text"] for sample in train_sentences],
        "review_rating": [sample["review_rating"] for sample in train_sentences],
    })
    
    test_dataset = Dataset.from_dict({
        "text": [sample["text"] for sample in test_sentences],
        "review_rating": [sample["review_rating"] for sample in test_sentences],
    })

    if len(train_sentences) == 0 or len(test_sentences) == 0:
        raise ValueError("Empty dataset after preprocessing/splitting")
    
    return {"train": train_dataset, "test": test_dataset}