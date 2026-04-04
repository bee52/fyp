from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "UK_Political_FakeNews_Detection", "random_seed": 42},
    "paths": {"raw_data_dir": "data/raw", "processed_data_dir": "data/processed"},
    "data_contract": {
        "required_columns": ["title", "text", "source", "date", "label"],
        "label_map": {
            "real": ["REAL", "real", "0", "0.0", "true"],
            "fake": ["FAKE", "fake", "satire", "1", "1.0", "false"],
        },
    },
    "splits": {"test_size": 0.2, "val_size": 0.1, "stratify": True},
    "scraping": {
        "run_mode": "standard",
        "default_output_dir": "data/raw",
        "real": {"max_articles_per_source": 200, "base_delay_seconds": 2.0, "max_retries": 3},
        "fake": {"max_articles_per_source": 200, "base_delay_seconds": 2.0, "max_retries": 3},
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = deepcopy(DEFAULT_CONFIG)

    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a top-level mapping")
        config = _deep_merge(config, loaded)

    return config


def project_path(relative_or_absolute: str | Path) -> Path:
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directory(path_like: str | Path) -> Path:
    directory = project_path(path_like)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
