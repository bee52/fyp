# Model Training Change 07: Semantic Branch Upgrade Path (RoBERTa)

## Current status
The semantic branch is currently implemented with TF-IDF + Logistic Regression to provide immediate, reproducible training progress.

## Why this is acceptable now
1. It validates full training/evaluation plumbing.
2. It enables quick ablation baselines before GPU-heavy transformer tuning.
3. It avoids blocking project momentum while the dual-branch framework stabilizes.

## Planned upgrade
Replace semantic pipeline in `src/training.py` with RoBERTa fine-tuning while preserving:
- split loading contract,
- artifact output contract,
- metrics schema in `training_metrics.json`.

## Report statement template
"We first established a reproducible semantic baseline using sparse text representations, then upgraded Branch A to RoBERTa under the same data split and reporting protocol to isolate architecture gains from infrastructure changes."
