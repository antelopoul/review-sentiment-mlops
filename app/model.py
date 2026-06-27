from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import os
import torch


class SentimentClassifier:
    """
    Fine-tune a pretrained language model for sentence-level sentiment classification.
    Supports 3 classes: negative (0), neutral (1), positive (2).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_labels: int = 3,
        max_length: int = 128,
        output_dir: str = "./models/sentiment",
    ):
        """
        Initialize the sentiment classifier.

        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of classification labels (default 3: neg, neutral, pos)
            max_length: Maximum token length for truncation
            output_dir: Directory to save fine-tuned model
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.output_dir = output_dir

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )

    def tokenize_function(self, examples):
        """Tokenize text examples."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

    def prepare_datasets(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        """
        Tokenize train and test datasets.

        Args:
            train_dataset: Training dataset from preprocess.prepare_sentence_datasets()
            val_dataset: Validation dataset from preprocess.prepare_sentence_datasets()
            test_dataset: Test dataset from preprocess.prepare_sentence_datasets()

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) tokenized
        """
        train_tokenized = train_dataset.map(self.tokenize_function, batched=True)
        val_tokenized = val_dataset.map(self.tokenize_function, batched=True)
        test_tokenized = test_dataset.map(self.tokenize_function, batched=True)

        return train_tokenized, val_tokenized, test_tokenized

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        eval_strategy: str = "epoch",
        use_mlflow: bool = False,
    ):
        """
        Fine-tune the model on the training dataset.

        Args:
            train_dataset: Training dataset (already tokenized)
            val_dataset: Validation dataset (already tokenized)
            test_dataset: Test dataset (already tokenized)
            num_epochs: Number of training epochs
            batch_size: Batch size for training and evaluation
            learning_rate: Learning rate for optimizer
            eval_strategy: Evaluation strategy ('epoch', 'steps', or 'no')
            use_mlflow: Whether to log to MLflow
        """

        import evaluate
        import numpy as np

        # Load metrics from the Hugging Face evaluate suite
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        # Define the metric parsing callback function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # Convert raw logits probabilities into class index predictions (0, 1, or 2)
            predictions = np.argmax(logits, axis=-1)
            
            # Compute accuracy
            acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            # Compute macro-averaged F1-Score (handles 3 classes neutrally)
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
            
            return {"accuracy": acc, "f1_macro": f1}
        # Prepare tokenized datasets
        train_tokenized, val_tokenized, test_tokenized = self.prepare_datasets(train_dataset, val_dataset, test_dataset)

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy=eval_strategy,
            save_strategy=eval_strategy,
            learning_rate=learning_rate,
            weight_decay=0.01,
            seed=42,
            logging_steps=100,
            save_total_limit=2,
            report_to=["mlflow"] if use_mlflow else [],
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            processing_class=self.tokenizer, 
            compute_metrics=compute_metrics,
        )

        # Train with MLflow logging if enabled
        if use_mlflow:
            import mlflow
            print("Logging to MLflow...")
            mlflow.set_tag("model_type", "sentiment_classifier")
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("num_labels", self.num_labels)
            mlflow.log_param("max_length", self.max_length)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("eval_strategy", eval_strategy)
            mlflow.log_param("train_samples", len(train_tokenized))
            mlflow.log_param("test_samples", len(test_tokenized))

        print("Starting fine-tuning with MLflow tracking...")
        trainer.train()

        if use_mlflow:
            val_metrics = trainer.evaluate(
                eval_dataset=val_tokenized,
                metric_key_prefix="eval",
            )
            test_metrics = trainer.evaluate(
                eval_dataset=test_tokenized,
                metric_key_prefix="test",
            )
            mlflow.log_metrics(val_metrics)
            mlflow.log_metrics(test_metrics)
        
        print(f"Model saved to {self.output_dir}")
        return trainer

    def save(self, path: str = None):
        """Save the fine-tuned model and tokenizer."""
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}")

    def predict(self, texts: list):
        """
        Predict sentiment for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of predictions (label indices)
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if was_training:
            self.model.train()

        predictions = outputs.logits.argmax(dim=-1).tolist()
        return predictions
