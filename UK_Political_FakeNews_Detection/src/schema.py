from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

CANONICAL_COLUMNS = ["title", "text", "source", "date", "label"]


def normalize_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"0", "0.0", "real", "true"}:
        return 0
    if text in {"1", "1.0", "fake", "satire", "false"}:
        return 1
    raise ValueError(f"Unsupported label value: {value}")


def normalize_article_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "title": str(record.get("title", "")).strip(),
        "text": str(record.get("text", "")).strip(),
        "source": str(record.get("source", "")).strip(),
        "date": str(record.get("date", "")).strip(),
        "label": normalize_label(record.get("label", "")),
    }
    return normalized


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame = frame[CANONICAL_COLUMNS]
    frame["title"] = frame["title"].astype(str).str.strip()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["source"] = frame["source"].astype(str).str.strip()
    frame["date"] = frame["date"].astype(str).str.strip()
    frame["label"] = frame["label"].apply(normalize_label)

    return frame


def enforce_columns(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    normalized_records = [normalize_article_record(record) for record in records]
    return pd.DataFrame(normalized_records, columns=CANONICAL_COLUMNS)
