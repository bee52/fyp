from __future__ import annotations

from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score


def _evaluate_fusion(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": report,
    }


def train_sklearn_fusion(
    style_model,
    semantic_model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Fusion for scikit-learn branches: uses branch confidence scores (probabilities).
    Fusion inputs: [style_proba_class1, semantic_proba_class1]
    """
    style_scores = style_model.predict_proba(train_df[["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]])[:, 1]
    semantic_scores = semantic_model.predict_proba(train_df["text"].astype(str))[:, 1]

    fusion_X = np.column_stack([style_scores, semantic_scores])
    fusion_y = train_df["label"].to_numpy()

    fusion_model = LogisticRegression(max_iter=1000, random_state=42)
    fusion_model.fit(fusion_X, fusion_y)

    val_style = style_model.predict_proba(val_df[["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]])[:, 1]
    val_sem = semantic_model.predict_proba(val_df["text"].astype(str))[:, 1]
    val_X = np.column_stack([val_style, val_sem])
    val_pred = np.asarray(fusion_model.predict(val_X))

    metrics = _evaluate_fusion(val_df["label"], val_pred)
    return fusion_model, metrics


def evaluate_sklearn_fusion_on_test(
    style_model,
    semantic_model,
    fusion_model: LogisticRegression,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate scikit-learn fusion on test set using branch probabilities."""
    style_features_cols = ["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]
    y_test = test_df["label"]

    test_style_scores = style_model.predict_proba(test_df[style_features_cols])[:, 1]
    test_sem_scores = semantic_model.predict_proba(test_df["text"].astype(str))[:, 1]
    fusion_X = np.column_stack([test_style_scores, test_sem_scores])
    fusion_pred = np.asarray(fusion_model.predict(fusion_X))

    return _evaluate_fusion(y_test, fusion_pred)


def train_roberta_fusion(
    roberta_encoder,
    style_model,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str = "cpu",
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Fusion for RoBERTa + stylistic: uses RoBERTa embeddings + style confidence scores.
    Fusion inputs: [style_proba_class1, roberta_embedding_768d]
    """
    train_embeddings = roberta_encoder.encode(train_df["text"].astype(str).tolist(), device=device, convert_to_numpy=True)
    style_scores = style_model.predict_proba(train_df[["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]])[:, 1]

    fusion_X = np.column_stack([style_scores[:, np.newaxis], train_embeddings])
    fusion_y = train_df["label"].to_numpy()

    fusion_model = LogisticRegression(max_iter=1000, random_state=42)
    fusion_model.fit(fusion_X, fusion_y)

    val_embeddings = roberta_encoder.encode(val_df["text"].astype(str).tolist(), device=device, convert_to_numpy=True)
    val_style = style_model.predict_proba(val_df[["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]])[:, 1]
    val_X = np.column_stack([val_style[:, np.newaxis], val_embeddings])
    val_pred = np.asarray(fusion_model.predict(val_X))

    metrics = _evaluate_fusion(val_df["label"], val_pred)
    return fusion_model, metrics


def evaluate_roberta_fusion_on_test(
    roberta_encoder,
    style_model,
    fusion_model: LogisticRegression,
    test_df: pd.DataFrame,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate RoBERTa+style fusion on test set."""
    y_test = test_df["label"]

    test_embeddings = roberta_encoder.encode(test_df["text"].astype(str).tolist(), device=device, convert_to_numpy=True)
    test_style_scores = style_model.predict_proba(test_df[["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]])[:, 1]
    fusion_X = np.column_stack([test_style_scores[:, np.newaxis], test_embeddings])
    fusion_pred = np.asarray(fusion_model.predict(fusion_X))

    return _evaluate_fusion(y_test, fusion_pred)
