from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import re
from textblob import TextBlob

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from .config import load_config, project_path
    from .preprocessing import (
        STYLE_FEATURE_COLUMNS,
        add_style_features,
        balance_dataset,
        clean_text_column,
        load_and_harmonize,
        stratified_split,
        write_phase1_artifacts,
    )
    from .training import run_training_from_phase1
except ImportError:
    from config import load_config, project_path
    from preprocessing import (
        STYLE_FEATURE_COLUMNS,
        add_style_features,
        balance_dataset,
        clean_text_column,
        load_and_harmonize,
        stratified_split,
        write_phase1_artifacts,
    )
    from training import run_training_from_phase1


class UKFakeNewsPipeline:
    def __init__(
        self,
        config_path: str | Path | None = None,
        default_stack: str = "sklearn",
        sklearn_model_dir: str | Path = "models/phase2_fixed",
        roberta_model_dir: str | Path = "models/phase2_roberta",
        device: str = "cpu",
    ) -> None:
        print("Initializing Dual-Branch Pipeline...")
        self.config = load_config(config_path)

        self.default_stack = default_stack.lower().strip()
        self.device = device

        self.sklearn_model_dir = project_path(sklearn_model_dir)
        self.roberta_model_dir = project_path(roberta_model_dir)

        self._scikit_loaded = False
        self._roberta_loaded = False

        self.style_model: Any = None
        self.semantic_model: Any = None
        self.fusion_model: Any = None

        self.roberta_encoder: Any = None
        self.roberta_classifier: Any = None
        self.fusion_roberta_model: Any = None

        print("Pipeline initialized. Models will be lazy-loaded on first prediction.")

    def _load_sklearn_models(self) -> None:
        if self._scikit_loaded:
            return

        style_path = self.sklearn_model_dir / "branch_b_style_model.joblib"
        semantic_path = self.sklearn_model_dir / "branch_a_semantic_model.joblib"
        fusion_path = self.sklearn_model_dir / "fusion_model.joblib"

        missing = [p for p in [style_path, semantic_path, fusion_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing sklearn inference artifacts: " + ", ".join(str(p) for p in missing)
            )

        self.style_model = joblib.load(style_path)
        self.semantic_model = joblib.load(semantic_path)
        self.fusion_model = joblib.load(fusion_path)
        self._scikit_loaded = True

    def _load_roberta_models(self) -> None:
        if self._roberta_loaded:
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required for roberta stack inference.")

        from sentence_transformers import SentenceTransformer as ST

        style_path = self.roberta_model_dir / "branch_b_style_model.joblib"
        encoder_path = self.roberta_model_dir / "branch_a_roberta_encoder"
        classifier_path = self.roberta_model_dir / "branch_a_roberta_classifier.joblib"
        fusion_path = self.roberta_model_dir / "fusion_roberta_model.joblib"

        missing = [p for p in [style_path, encoder_path, fusion_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing roberta inference artifacts: " + ", ".join(str(p) for p in missing)
            )

        self.style_model = joblib.load(style_path)
        self.roberta_encoder = ST(str(encoder_path), device=self.device)
        self.fusion_roberta_model = joblib.load(fusion_path)

        if classifier_path.exists():
            self.roberta_classifier = joblib.load(classifier_path)
        else:
            self.roberta_classifier = None

        self._roberta_loaded = True

    def extract_branch_b_features(self, raw_text: str) -> Dict[str, float]:
        """Extract stylistic features with the same schema used during training."""
        text = str(raw_text or "").strip()
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return {
                "word_count": 0.0,
                "shout_ratio": 0.0,
                "exclamation_density": 0.0,
                "question_density": 0.0,
                "lexical_diversity": 0.0,
                "sentiment": 0.0,
            }

        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        lexical_diversity = len(set(words)) / word_count

        return {
            "word_count": float(word_count),
            "shout_ratio": float(len(caps_words) / word_count),
            "exclamation_density": float(text.count("!") / word_count),
            "question_density": float(text.count("?") / word_count),
            "lexical_diversity": float(lexical_diversity),
            "sentiment": float(getattr(TextBlob(text).sentiment, "polarity", 0.0)),
        }

    def _style_df(self, style_features: Dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame([style_features])[STYLE_FEATURE_COLUMNS]

    def prepare_branch_a_text(self, raw_text: str) -> str:
        """Clean text for semantic branch inference."""
        clean_text = str(raw_text or "")
        clean_text = re.sub(r"http\S+|www\S+|https\S+", "", clean_text, flags=re.MULTILINE)
        clean_text = re.sub(r"\s+", " ", clean_text).strip().lower()
        return clean_text

    def predict(self, raw_text: str, stack: str | None = None) -> Dict[str, Any]:
        """Run inference using either 'sklearn' or 'roberta' stack."""
        selected_stack = (stack or self.default_stack).lower().strip()
        if selected_stack not in {"sklearn", "roberta"}:
            raise ValueError("stack must be one of: sklearn, roberta")

        if str(raw_text or "").strip() == "":
            raise ValueError("raw_text cannot be empty")

        style_features = self.extract_branch_b_features(raw_text)
        clean_text = self.prepare_branch_a_text(raw_text)
        style_input = self._style_df(style_features)

        if selected_stack == "sklearn":
            self._load_sklearn_models()
            assert self.style_model is not None
            assert self.semantic_model is not None
            assert self.fusion_model is not None
            style_score = float(self.style_model.predict_proba(style_input)[0, 1])
            semantic_score = float(self.semantic_model.predict_proba([clean_text])[0, 1])

            fusion_input = np.array([[style_score, semantic_score]], dtype=float)
            fake_probability = float(self.fusion_model.predict_proba(fusion_input)[0, 1])

            branch_scores: Dict[str, float | None] = {
                "style_fake_probability": style_score,
                "semantic_fake_probability": semantic_score,
                "fusion_fake_probability": fake_probability,
            }
        else:
            self._load_roberta_models()
            assert self.style_model is not None
            assert self.roberta_encoder is not None
            assert self.fusion_roberta_model is not None
            style_score = float(self.style_model.predict_proba(style_input)[0, 1])

            embedding = self.roberta_encoder.encode(
                [clean_text],
                device=self.device,
                convert_to_numpy=True,
            )
            fusion_input = np.column_stack([np.array([style_score]), embedding])
            fake_probability = float(self.fusion_roberta_model.predict_proba(fusion_input)[0, 1])

            semantic_score = None
            if self.roberta_classifier is not None:
                semantic_score = float(self.roberta_classifier.predict_proba(embedding)[0, 1])

            branch_scores = {
                "style_fake_probability": style_score,
                "semantic_fake_probability": semantic_score,
                "fusion_fake_probability": fake_probability,
            }

        prediction = "Unreliable" if fake_probability >= 0.5 else "Reliable"
        confidence = fake_probability if prediction == "Unreliable" else (1.0 - fake_probability)

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_probability),
            "stack": selected_stack,
            "branch_scores": branch_scores,
            "stylistic_breakdown": style_features,
        }

    def run_phase1_data_preparation(self, real_csv: str | Path, fake_csv: str | Path) -> Dict[str, Any]:
        """Builds deterministic Phase 1 artifacts from canonical raw sources."""
        seed = int(self.config["project"]["random_seed"])
        split_cfg = self.config["splits"]
        processed_dir = project_path(self.config["paths"]["processed_data_dir"]) / "phase1"

        combined = load_and_harmonize(real_csv, fake_csv)
        balanced = balance_dataset(combined, random_seed=seed)
        featured = add_style_features(balanced, text_column="text")  # BEFORE cleaning
        cleaned = clean_text_column(featured, text_column="text")     # AFTER features

        train_df, val_df, test_df = stratified_split(
            cleaned,
            test_size=float(split_cfg["test_size"]),
            val_size=float(split_cfg["val_size"]),
            random_seed=seed,
        )
        artifact_paths = write_phase1_artifacts(train_df, val_df, test_df, processed_dir, random_seed=seed)

        return {
            "seed": seed,
            "input": {
                "real_csv": str(Path(real_csv)),
                "fake_csv": str(Path(fake_csv)),
            },
            "rows": {
                "combined": int(len(combined)),
                "balanced": int(len(balanced)),
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "artifacts": {k: str(v) for k, v in artifact_paths.items()},
        }

    def run_model_training(
        self,
        split_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Runs Branch B, Branch A semantic baseline, and fusion model training."""
        training_cfg = self.config.get("training", {})
        split_path = split_dir or training_cfg.get("split_dir", "data/processed/phase1")
        output_path = output_dir or training_cfg.get("output_dir", "models/phase2")

        return run_training_from_phase1(split_dir=split_path, model_output_dir=output_path)