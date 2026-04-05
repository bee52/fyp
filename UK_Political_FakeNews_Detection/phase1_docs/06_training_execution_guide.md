# Model Training Change 06: Execution Guide

## Command to train all branches + fusion
From project root:

```bash
python -m src.train_models --split-dir data/processed/phase1 --output-dir models/phase2
```

## Expected outputs
Inside `models/phase2/`:
1. `branch_b_style_model.joblib`
2. `branch_a_semantic_model.joblib`
3. `fusion_model.joblib`
4. `training_metrics.json`

## Preconditions
1. `data/processed/phase1/train.csv` exists.
2. `data/processed/phase1/val.csv` exists.
3. `data/processed/phase1/test.csv` exists.
4. Phase 1 preprocessing has already been run.

## Recommended report screenshots
1. Terminal output of training command.
2. Metrics excerpt from `training_metrics.json` showing macro-F1 for all three variants.
3. Model artifact folder contents.
