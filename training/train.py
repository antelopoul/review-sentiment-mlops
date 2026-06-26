from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "training" / "config.yaml"

import mlflow
from app.model import SentimentClassifier
from transformers import set_seed

# Set random seed for reproducibility
set_seed(42)

if __name__ == "__main__":
    from training.preprocess import prepare_sentence_datasets
    from training.config_loader import load_config

    # Load configuration
    config = load_config(str(CONFIG_PATH))
    
    # Set up MLflow tracking backend FIRST
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sentiment-classification")
    
    # Now you can safely start logging your run parameters
    with mlflow.start_run():
        mlflow.log_params(config["model"])
        mlflow.log_params(config["training"])
    
        print("=" * 70)
        print("Sentiment Classifier Training with MLflow")
        print("=" * 70)
        print(f"Config: {CONFIG_PATH}")
        print(f"Model: {config['model']['model_name']}")
        print(f"Data: {config['data']['data_files']}")
        print(f"MLflow Experiment: sentiment-classification")
        print(f"MLflow Tracking URI: sqlite:///mlflow.db")
        print("=" * 70)

        # Load and prepare datasets
        splits = prepare_sentence_datasets(
            data_files=config["data"]["data_files"],
            val_size=config["data"]["val_size"], 
            test_size=config["data"]["test_size"],
            seed=config["data"]["seed"],
            label_map=config["label_map"],
            spacy_model=config["preprocessing"]["spacy_model"],
            disable_components=config["preprocessing"]["disable_components"],
            min_sentence_length=config["preprocessing"]["min_sentence_length"]
        )

        train_dataset = splits["train"]
        val_dataset = splits["val"]
        test_dataset = splits["test"]

        print(f"\nTrain: {len(train_dataset)} samples")
        print(f"Validation: {len(val_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples\n")

        # Initialize classifier and train
        classifier = SentimentClassifier(
            model_name=config["model"]["model_name"],
            num_labels=config["model"]["num_labels"],
            max_length=config["model"]["max_length"],
            output_dir=config["model"]["output_dir"],
        )

        # Train the model (with MLflow logging enabled)
        trainer = classifier.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_epochs=config["training"]["num_epochs"],
            batch_size=config["training"]["per_device_train_batch_size"],
            learning_rate=config["training"]["learning_rate"],
            eval_strategy=config["training"]["eval_strategy"],
            use_mlflow=True,  # Enable MLflow logging
        )

        # Save model
        classifier.save()

    # Example inference
    print("\n" + "=" * 70)
    print("Example Inference")
    print("=" * 70)
    test_texts = [
        "This book is amazing and I loved every page.",
        "It was okay, nothing special.",
        "Terrible quality, waste of money.",
    ]
    predictions = classifier.predict(test_texts)
    labels = ["negative", "neutral", "positive"]
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}\n-> {labels[pred]}\n")
    
    print("=" * 70)
    print("Training complete! View results with: mlflow ui")
    print("=" * 70)