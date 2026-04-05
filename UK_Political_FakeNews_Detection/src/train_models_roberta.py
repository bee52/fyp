from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .training import (
        load_phase1_splits,
        train_stylistic_branch,
        _evaluate_binary,
    )
    from .training_roberta import (
        train_roberta_semantic_branch,
        evaluate_roberta_on_test,
        save_roberta_artifacts,
    )
    from .fusion import train_roberta_fusion, evaluate_roberta_fusion_on_test
    from .config import ensure_directory
except ImportError:
    from training import (
        load_phase1_splits,
        train_stylistic_branch,
        _evaluate_binary,
    )
    from training_roberta import (
        train_roberta_semantic_branch,
        evaluate_roberta_on_test,
        save_roberta_artifacts,
    )
    from fusion import train_roberta_fusion, evaluate_roberta_fusion_on_test
    from config import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Branch B (stylistic), Branch A (RoBERTa), and Fusion")
    parser.add_argument("--split-dir", default="data/processed/phase1", help="Directory with train/val/test CSVs")
    parser.add_argument("--output-dir", default="models/phase2_roberta", help="Output directory for models")
    parser.add_argument("--roberta-model", default="distilroberta-base", help="HuggingFace model ID")
    parser.add_argument("--device", default="cpu", help="Device for RoBERTa (cpu, cuda, mps)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load splits
    train_df, val_df, test_df = load_phase1_splits(args.split_dir)

    # Train Branch B (stylistic - RandomForest)
    style_model, style_val_metrics = train_stylistic_branch(train_df, val_df)

    # Train Branch A (semantic - RoBERTa)
    print("Loading RoBERTa encoder...")
    roberta_encoder, roberta_classifier, roberta_val_metrics = train_roberta_semantic_branch(
        train_df, val_df, model_name=args.roberta_model, device=args.device
    )

    # Train Fusion (style + RoBERTa embeddings)
    print("Training fusion layer...")
    fusion_model, fusion_val_metrics = train_roberta_fusion(
        roberta_encoder, style_model, train_df, val_df, device=args.device
    )

    # Evaluate on test set
    import numpy as np
    style_features = ["word_count", "shout_ratio", "exclamation_density", "question_density", "lexical_diversity", "sentiment"]
    style_test_pred = np.asarray(style_model.predict(test_df[style_features]))
    style_test_metrics = _evaluate_binary(test_df["label"], style_test_pred)
    
    roberta_test_metrics = evaluate_roberta_on_test(roberta_encoder, roberta_classifier, test_df, device=args.device)
    fusion_test_metrics = evaluate_roberta_fusion_on_test(roberta_encoder, style_model, fusion_model, test_df, device=args.device)

    # Save style model and fusion  
    out_dir = ensure_directory(args.output_dir)
    import joblib
    joblib.dump(style_model, out_dir / "branch_b_style_model.joblib")
    joblib.dump(fusion_model, out_dir / "fusion_roberta_model.joblib")

    all_metrics = {
        "validation": {
            "style": style_val_metrics,
            "roberta": roberta_val_metrics,
            "fusion": fusion_val_metrics,
        },
        "test": {
            "style_test": style_test_metrics,
            "roberta_test": roberta_test_metrics,
            "fusion_roberta_test": fusion_test_metrics,
        },
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "config": {
            "roberta_model": args.roberta_model,
            "device": args.device,
        },
    }

    roberta_artifacts = save_roberta_artifacts(args.output_dir, roberta_encoder, roberta_classifier)

    with (out_dir / "training_metrics_roberta.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    result = {
        "metrics": all_metrics,
        "artifacts": {
            "style_model": str(out_dir / "branch_b_style_model.joblib"),
            "roberta_encoder": roberta_artifacts["encoder"],
            "roberta_classifier": roberta_artifacts["classifier"],
            "fusion_model": str(out_dir / "fusion_roberta_model.joblib"),
            "metrics": str(out_dir / "training_metrics_roberta.json"),
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
