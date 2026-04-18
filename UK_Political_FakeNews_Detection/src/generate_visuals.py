from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required. Install dependencies from requirements.txt first."
    ) from exc

try:
    from .preprocessing import STYLE_FEATURE_COLUMNS, extract_style_features
    from .schema import normalize_label
except ImportError:
    from preprocessing import STYLE_FEATURE_COLUMNS, extract_style_features
    from schema import normalize_label


sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Table 5.2 and Figures 5.2/5.3 from the gold-standard stress test dataset."
    )
    parser.add_argument(
        "--gold-csv",
        default=str(PROJECT_ROOT / "data" / "raw" / "uk_fake_manual.csv"),
        help="Path to manually curated fake-only gold-standard CSV.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "phase2_roberta"),
        help="Directory containing RoBERTa branch and fusion artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "visuals"),
        help="Directory for generated table and figure assets.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for SentenceTransformer encoder (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--save-audit-csv",
        action="store_true",
        help="Save row-level prediction audit CSV for traceability.",
    )
    return parser.parse_args()


def _semantic_clean(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"http\S+|www\S+|https\S+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def load_gold_dataset(gold_csv: Path) -> pd.DataFrame:
    if not gold_csv.exists():
        raise FileNotFoundError(f"Gold dataset not found: {gold_csv}")

    frame = None
    for encoding in ["utf-8", "cp1252", "latin-1"]:
        try:
            frame = pd.read_csv(gold_csv, encoding=encoding, engine="python", on_bad_lines="skip")
            break
        except UnicodeDecodeError:
            continue

    if frame is None:
        raise ValueError(
            "Unable to parse gold CSV with supported encodings (utf-8, cp1252, latin-1)."
        )

    if "title" not in frame.columns:
        frame["title"] = ""
    if "text" not in frame.columns:
        frame["text"] = ""
    if "label" not in frame.columns:
        raise ValueError("Gold CSV must include a 'label' column.")

    frame["title"] = frame["title"].fillna("").astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)

    non_empty_mask = ~(
        frame["title"].str.strip().eq("") & frame["text"].str.strip().eq("")
    )
    cleaned = frame.loc[non_empty_mask].copy()

    cleaned["label"] = cleaned["label"].apply(normalize_label)
    cleaned["semantic_text"] = cleaned["text"].apply(_semantic_clean)
    cleaned["style_source_text"] = cleaned["text"].astype(str)

    return cleaned.reset_index(drop=True)


def build_style_features(df: pd.DataFrame) -> pd.DataFrame:
    style_rows = []
    for raw_text in df["style_source_text"].tolist():
        try:
            style_rows.append(extract_style_features(str(raw_text)))
        except Exception:
            # Fall back to zeros if one malformed row fails feature extraction.
            style_rows.append(
                {
                    "word_count": 0.0,
                    "shout_ratio": 0.0,
                    "exclamation_density": 0.0,
                    "question_density": 0.0,
                    "lexical_diversity": 0.0,
                    "sentiment": 0.0,
                }
            )

    style_df = pd.DataFrame(style_rows)
    return style_df[STYLE_FEATURE_COLUMNS]


def load_roberta_stack(model_dir: Path, device: str):
    style_path = model_dir / "branch_b_style_model.joblib"
    encoder_path = model_dir / "branch_a_roberta_encoder"
    branch_a_path = model_dir / "branch_a_roberta_classifier.joblib"
    fusion_path = model_dir / "fusion_roberta_model.joblib"

    missing = [p for p in [style_path, encoder_path, branch_a_path, fusion_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required model artifacts: " + ", ".join(str(p) for p in missing))

    style_model = joblib.load(style_path)
    encoder = SentenceTransformer(str(encoder_path), device=device)
    branch_a_classifier = joblib.load(branch_a_path)
    fusion_model = joblib.load(fusion_path)
    return style_model, encoder, branch_a_classifier, fusion_model


def evaluate_fake_class(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision[0]),
        "recall_fake": float(recall[0]),
        "f1_fake": float(f1[0]),
        "support_fake": int(support[0]),
        "predicted_fake_rate": float(np.mean(y_pred == 1)),
    }


def render_table_5_2(table_df: pd.DataFrame, output_png: Path) -> None:
    display_df = table_df.copy()
    for col in ["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1 (Fake)", "Predicted Fake Rate"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Table 5.2 - Gold Standard Stress Test Evaluation Metrics", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def render_figure_5_3_f1(table_df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=table_df,
        x="Model",
        y="F1 (Fake)",
        hue="Model",
        palette=["#4C78A8", "#F58518"],
        legend=False,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1 Score (Fake Class)")
    ax.set_title("Figure 5.3 - Branch A vs Hybrid Fusion (Gold Stress Test)")

    for patch in ax.patches:
        value = patch.get_height()
        ax.annotate(
            f"{value:.4f}",
            (patch.get_x() + patch.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def render_figure_5_2_gini(style_model, output_png: Path) -> None:
    try:
        gini = style_model.named_steps["clf"].feature_importances_
    except Exception as exc:
        raise ValueError("Could not extract Branch B RandomForest feature importances.") from exc

    importance_df = pd.DataFrame(
        {
            "Feature": STYLE_FEATURE_COLUMNS,
            "Gini Importance": gini,
        }
    ).sort_values("Gini Importance", ascending=True)

    plt.figure(figsize=(9, 5.5))
    ax = sns.barplot(
        data=importance_df,
        x="Gini Importance",
        y="Feature",
        hue="Feature",
        palette="Reds_r",
        legend=False,
    )
    ax.set_title("Figure 5.2 - Gini Feature Importance (Branch B)")
    ax.set_xlabel("Mean Decrease in Impurity (Normalized)")
    ax.set_ylabel("Stylistic Feature")

    for patch in ax.patches:
        value = patch.get_width()
        ax.annotate(
            f"{value:.4f}",
            (value, patch.get_y() + patch.get_height() / 2),
            ha="left",
            va="center",
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def build_table_rows(
    y_true: np.ndarray,
    branch_a_pred: np.ndarray,
    fusion_pred: np.ndarray,
) -> pd.DataFrame:
    branch_a_metrics = evaluate_fake_class(y_true, branch_a_pred)
    fusion_metrics = evaluate_fake_class(y_true, fusion_pred)

    return pd.DataFrame(
        [
            {
                "Model": "Branch A (RoBERTa)",
                "Accuracy": branch_a_metrics["accuracy"],
                "Precision (Fake)": branch_a_metrics["precision_fake"],
                "Recall (Fake)": branch_a_metrics["recall_fake"],
                "F1 (Fake)": branch_a_metrics["f1_fake"],
                "Support (Fake)": branch_a_metrics["support_fake"],
                "Predicted Fake Rate": branch_a_metrics["predicted_fake_rate"],
            },
            {
                "Model": "Hybrid Fusion",
                "Accuracy": fusion_metrics["accuracy"],
                "Precision (Fake)": fusion_metrics["precision_fake"],
                "Recall (Fake)": fusion_metrics["recall_fake"],
                "F1 (Fake)": fusion_metrics["f1_fake"],
                "Support (Fake)": fusion_metrics["support_fake"],
                "Predicted Fake Rate": fusion_metrics["predicted_fake_rate"],
            },
        ]
    )


def generate_assets(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    gold_csv = Path(args.gold_csv)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_gold_dataset(gold_csv)
    if df.empty:
        raise ValueError("No valid rows in gold dataset after blank-row filtering.")

    y_true = df["label"].to_numpy()
    style_df = build_style_features(df)

    style_model, encoder, branch_a_classifier, fusion_model = load_roberta_stack(
        model_dir=model_dir,
        device=args.device,
    )

    texts = df["semantic_text"].tolist()
    embeddings = encoder.encode(texts, device=args.device, convert_to_numpy=True)

    branch_a_pred = np.asarray(branch_a_classifier.predict(embeddings))
    style_probs = style_model.predict_proba(style_df)[:, 1]
    fusion_X = np.column_stack([style_probs[:, np.newaxis], embeddings])
    fusion_pred = np.asarray(fusion_model.predict(fusion_X))

    table_df = build_table_rows(y_true, branch_a_pred, fusion_pred)

    table_csv_path = output_dir / "table_5_2_stress_test_metrics.csv"
    table_png_path = output_dir / "table_5_2_stress_test_metrics.png"
    fig_5_2_path = output_dir / "figure_5_2_gini_importance.png"
    fig_5_3_path = output_dir / "figure_5_3_f1_comparison.png"

    table_df.to_csv(table_csv_path, index=False)
    render_table_5_2(table_df, table_png_path)
    render_figure_5_2_gini(style_model, fig_5_2_path)
    render_figure_5_3_f1(table_df, fig_5_3_path)

    if args.save_audit_csv:
        audit_df = df[["title", "text", "label"]].copy()
        audit_df["pred_branch_a"] = branch_a_pred
        audit_df["pred_hybrid_fusion"] = fusion_pred
        audit_df["prob_style_fake"] = style_probs
        audit_df.to_csv(output_dir / "gold_stress_test_predictions.csv", index=False)

    return table_csv_path, table_png_path, fig_5_2_path, fig_5_3_path


def main() -> None:
    args = parse_args()
    table_csv, table_png, fig_5_2, fig_5_3 = generate_assets(args)

    print("Generated assets:")
    print(f"- {table_csv}")
    print(f"- {table_png}")
    print(f"- {fig_5_2}")
    print(f"- {fig_5_3}")
    print("Note: This is a fake-only gold stress test; metrics are class-1 focused.")


if __name__ == "__main__":
    main()