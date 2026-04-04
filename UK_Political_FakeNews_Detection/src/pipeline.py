# src/pipeline.py
from pathlib import Path

import pandas as pd
from textblob import TextBlob
import re
import joblib # Used to load saved scikit-learn models
# from transformers import pipeline as hf_pipeline # Uncomment when RoBERTa is ready

try:
    from .config import load_config, project_path
    from .preprocessing import (
        add_style_features,
        balance_dataset,
        clean_text_column,
        load_and_harmonize,
        stratified_split,
        write_phase1_artifacts,
    )
except ImportError:
    from config import load_config, project_path
    from preprocessing import (
        add_style_features,
        balance_dataset,
        clean_text_column,
        load_and_harmonize,
        stratified_split,
        write_phase1_artifacts,
    )

class UKFakeNewsPipeline:
    def __init__(self, config_path=None):
        print("Initializing Dual-Branch Pipeline...")
        self.config = load_config(config_path)
        
        # Load Branch B (Stylistic Model - e.g., a Random Forest)
        # self.style_model = joblib.load('../models/stylistic_rf_model.pkl')
        
        # Load Branch A (Semantic Model - RoBERTa)
        # self.semantic_model = hf_pipeline("text-classification", model="../models/roberta_fine_tuned")
        
        print("Models loaded successfully.")

    def extract_branch_b_features(self, raw_text):
        """Extracts stylistic math from raw text."""
        words = raw_text.split()
        word_count = len(words)
        if word_count == 0:
            return {'word_count': 0, 'shout_ratio': 0.0, 'exclamations': 0.0, 'sentiment': 0.0}

        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        
        return {
            'word_count': word_count,
            'shout_ratio': len(caps_words) / word_count,
            'exclamations': raw_text.count('!') / word_count,
            'sentiment': float(getattr(TextBlob(raw_text).sentiment, 'polarity', 0.0))
        }

    def prepare_branch_a_text(self, raw_text):
        """Cleans text for RoBERTa."""
        clean_text = raw_text.lower()
        clean_text = re.sub(r'http\S+|www\S+|https\S+', '', clean_text, flags=re.MULTILINE)
        return clean_text

    def predict(self, raw_text):
        """The Master Inference Function"""
        # 1. Get Style Features
        style_features = self.extract_branch_b_features(raw_text)
        
        # 2. Get Clean Text
        clean_text = self.prepare_branch_a_text(raw_text)
        
        # --- FUTURE INTEGRATION (Once models are trained) ---
        # style_score = self.style_model.predict_proba(pd.DataFrame([style_features]))[0][1]
        # semantic_score = self.semantic_model(clean_text)[0]['score']
        # final_confidence = (style_score + semantic_score) / 2 # Simple fusion
        
        # For now, return mock scores so you can build the UI
        return {
            "prediction": "Unreliable",
            "confidence": 0.85,
            "stylistic_breakdown": style_features
        }

    def run_phase1_data_preparation(self, real_csv, fake_csv):
        """Builds deterministic Phase 1 artifacts from canonical raw sources."""
        seed = int(self.config["project"]["random_seed"])
        split_cfg = self.config["splits"]
        processed_dir = project_path(self.config["paths"]["processed_data_dir"]) / "phase1"

        combined = load_and_harmonize(real_csv, fake_csv)
        balanced = balance_dataset(combined, random_seed=seed)
        cleaned = clean_text_column(balanced, text_column="text")
        featured = add_style_features(cleaned, text_column="text")

        train_df, val_df, test_df = stratified_split(
            featured,
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