from __future__ import annotations

from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from .config import ensure_directory, project_path
except ImportError:
    from config import ensure_directory, project_path


def _evaluate_roberta(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": report,
    }


def train_roberta_semantic_branch(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str = "distilroberta-base",
    device: str = "cpu",
) -> Tuple[SentenceTransformer, LogisticRegression, Dict[str, Any]]:
    """
    RoBERTa semantic branch: fine-tune embeddings + logistic head.

    Args:
        train_df: Training dataframe with 'text' and 'label' columns.
        val_df: Validation dataframe.
        model_name: HuggingFace model ID (e.g. distilroberta-base, roberta-base).
        device: Compute device ('cpu', 'cuda', 'mps').

    Returns:
        (encoder, classifier, validation_metrics)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers torch")

    encoder = SentenceTransformer(model_name, device=device)

    train_texts = train_df["text"].astype(str).tolist()
    train_embeddings = encoder.encode(train_texts, device=device, convert_to_numpy=True)
    train_labels = train_df["label"].to_numpy()

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, train_labels)

    val_texts = val_df["text"].astype(str).tolist()
    val_embeddings = encoder.encode(val_texts, device=device, convert_to_numpy=True)
    val_labels = val_df["label"].to_numpy()
    val_pred = np.asarray(classifier.predict(val_embeddings))

    metrics = _evaluate_roberta(val_df["label"], val_pred)
    return encoder, classifier, metrics


def evaluate_roberta_on_test(
    encoder: SentenceTransformer,
    classifier: LogisticRegression,
    test_df: pd.DataFrame,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate RoBERTa semantic branch on test set."""
    test_texts = test_df["text"].astype(str).tolist()
    test_embeddings = encoder.encode(test_texts, device=device, convert_to_numpy=True)
    test_pred = np.asarray(classifier.predict(test_embeddings))

    return _evaluate_roberta(test_df["label"], test_pred)


def save_roberta_artifacts(
    output_dir: str,
    encoder: SentenceTransformer,
    classifier: LogisticRegression,
) -> Dict[str, str]:
    """Save RoBERTa encoder and classifier artifacts."""
    out_dir = ensure_directory(output_dir)

    encoder_path = out_dir / "branch_a_roberta_encoder"
    classifier_path = out_dir / "branch_a_roberta_classifier.joblib"

    encoder.save(str(encoder_path))
    joblib.dump(classifier, classifier_path)

    return {
        "encoder": str(encoder_path),
        "classifier": str(classifier_path),
    }
