from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .config import ensure_directory, project_path
    from .preprocessing import STYLE_FEATURE_COLUMNS
    from .fusion import train_sklearn_fusion, evaluate_sklearn_fusion_on_test
except ImportError:
    from config import ensure_directory, project_path
    from preprocessing import STYLE_FEATURE_COLUMNS
    from fusion import train_sklearn_fusion, evaluate_sklearn_fusion_on_test


def _read_split_csv(split_dir: Path, name: str) -> pd.DataFrame:
    path = split_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def load_phase1_splits(split_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_path = project_path(split_dir)
    train_df = _read_split_csv(split_path, "train")
    val_df = _read_split_csv(split_path, "val")
    test_df = _read_split_csv(split_path, "test")
    return train_df, val_df, test_df


def _evaluate_binary(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": report,
    }


def train_stylistic_branch(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train RandomForest on engineered style features (Branch B)."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ])

    X_train = train_df[STYLE_FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_val = val_df[STYLE_FEATURE_COLUMNS]
    y_val = val_df["label"]

    model.fit(X_train, y_train)
    val_pred = np.asarray(model.predict(X_val))
    metrics = _evaluate_binary(y_val, val_pred)
    return model, metrics


def train_semantic_branch(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    max_features: int = 20000
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train TF-IDF + LogisticRegression baseline on text (Branch A)."""
    semantic_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            ),
        ),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    X_train = train_df["text"].astype(str)
    y_train = train_df["label"]
    X_val = val_df["text"].astype(str)
    y_val = val_df["label"]

    semantic_pipeline.fit(X_train, y_train)
    val_pred = np.asarray(semantic_pipeline.predict(X_val))
    metrics = _evaluate_binary(y_val, val_pred)
    return semantic_pipeline, metrics


def train_fusion_model(
    style_model: Pipeline,
    semantic_model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """Wrapper around standalone fusion function for scikit stack."""
    return train_sklearn_fusion(style_model, semantic_model, train_df, val_df)


def evaluate_all_on_test(
    style_model: Pipeline,
    semantic_model: Pipeline,
    fusion_model: LogisticRegression,
    test_df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all branches on test set."""
    y_test = test_df["label"]

    style_pred = np.asarray(style_model.predict(test_df[STYLE_FEATURE_COLUMNS]))
    semantic_pred = np.asarray(semantic_model.predict(test_df["text"].astype(str)))
    fusion_test_metrics = evaluate_sklearn_fusion_on_test(style_model, semantic_model, fusion_model, test_df)

    return {
        "style_test": _evaluate_binary(y_test, style_pred),
        "semantic_test": _evaluate_binary(y_test, semantic_pred),
        "fusion_test": fusion_test_metrics,
    }


def save_training_artifacts(
    out_dir: str | Path,
    style_model: Pipeline,
    semantic_model: Pipeline,
    fusion_model: LogisticRegression,
    metrics: Dict[str, object],
) -> Dict[str, str]:
    """Save trained models and metrics to disk."""
    out_path = project_path(out_dir)
    ensure_directory(out_path)

    style_path = out_path / "branch_b_style_model.joblib"
    semantic_path = out_path / "branch_a_semantic_model.joblib"
    fusion_path = out_path / "fusion_model.joblib"
    metrics_path = out_path / "training_metrics.json"

    joblib.dump(style_model, style_path)
    joblib.dump(semantic_model, semantic_path)
    joblib.dump(fusion_model, fusion_path)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "style_model": str(style_path),
        "semantic_model": str(semantic_path),
        "fusion_model": str(fusion_path),
        "metrics": str(metrics_path),
    }


def run_training_from_phase1(
    split_dir: str | Path = "data/processed/phase1",
    model_output_dir: str | Path = "models/phase2",
) -> Dict[str, object]:
    """End-to-end training pipeline: load splits → train branches → fuse → eval test."""
    train_df, val_df, test_df = load_phase1_splits(split_dir)

    style_model, style_val_metrics = train_stylistic_branch(train_df, val_df)
    semantic_model, semantic_val_metrics = train_semantic_branch(train_df, val_df)
    fusion_model, fusion_val_metrics = train_fusion_model(style_model, semantic_model, train_df, val_df)

    test_metrics = evaluate_all_on_test(style_model, semantic_model, fusion_model, test_df)

    all_metrics: Dict[str, object] = {
        "validation": {
            "style": style_val_metrics,
            "semantic": semantic_val_metrics,
            "fusion": fusion_val_metrics,
        },
        "test": test_metrics,
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "features": {
            "style": STYLE_FEATURE_COLUMNS,
            "semantic": "tfidf_unigram_bigram",
            "fusion": "logistic_regression_on_branch_probabilities",
        },
    }

    artifacts = save_training_artifacts(model_output_dir, style_model, semantic_model, fusion_model, all_metrics)
    return {
        "metrics": all_metrics,
        "artifacts": artifacts,
    }
