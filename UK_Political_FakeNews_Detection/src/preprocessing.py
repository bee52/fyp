from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob

try:
    from .schema import normalize_dataframe
except ImportError:
    from schema import normalize_dataframe


STYLE_FEATURE_COLUMNS = [
    "word_count",
    "shout_ratio",
    "exclamation_density",
    "question_density",
    "lexical_diversity",
    "sentiment",
]


def load_and_harmonize(real_csv_path: str | Path, fake_csv_path: str | Path) -> pd.DataFrame:
    real_df = pd.read_csv(real_csv_path)
    fake_df = pd.read_csv(fake_csv_path)

    if "label" not in real_df.columns:
        real_df["label"] = 0
    if "label" not in fake_df.columns:
        fake_df["label"] = 1

    combined = pd.concat([real_df, fake_df], ignore_index=True)
    return normalize_dataframe(combined)


def balance_dataset(df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    normalized = normalize_dataframe(df)
    counts = normalized["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Dataset must contain both classes 0 and 1")

    minority_count = counts.min()
    balanced = (
        normalized.groupby("label", group_keys=False)
        .apply(lambda part: part.sample(n=minority_count, random_state=random_seed))
        .sample(frac=1.0, random_state=random_seed)
        .reset_index(drop=True)
    )
    return balanced


def clean_text_column(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    frame = df.copy()
    frame[text_column] = frame[text_column].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return frame


def extract_style_features(text: str) -> Dict[str, float]:
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return {
            "word_count": 0,
            "shout_ratio": 0.0,
            "exclamation_density": 0.0,
            "question_density": 0.0,
            "lexical_diversity": 0.0,
            "sentiment": 0.0,
        }

    caps_words = [word for word in words if word.isupper() and len(word) > 1]
    lexical_diversity = len(set(words)) / word_count

    return {
        "word_count": float(word_count),
        "shout_ratio": len(caps_words) / word_count,
        "exclamation_density": text.count("!") / word_count,
        "question_density": text.count("?") / word_count,
        "lexical_diversity": lexical_diversity,
        "sentiment": float(getattr(TextBlob(text).sentiment, "polarity", 0.0)),
    }


def add_style_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    frame = df.copy()
    features = frame[text_column].astype(str).apply(extract_style_features).apply(pd.Series)
    return pd.concat([frame, features], axis=1)


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    normalized = normalize_dataframe(df)

    train_val, test = train_test_split(
        normalized,
        test_size=test_size,
        random_state=random_seed,
        stratify=normalized["label"],
    )

    val_fraction_of_train_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=random_seed,
        stratify=train_val["label"],
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def write_phase1_artifacts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    random_seed: int,
) -> Dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    metadata_path = out_dir / "split_metadata.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": random_seed,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_class_counts": train_df["label"].value_counts().to_dict(),
        "val_class_counts": val_df["label"].value_counts().to_dict(),
        "test_class_counts": test_df["label"].value_counts().to_dict(),
    }
    pd.Series(metadata).to_json(metadata_path, indent=2)

    return {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "metadata": metadata_path,
    }
