# Model Training Run Results (2026-04-04)

## Run sequence
1. Generated Phase 1 artifacts with:
   - `python -m src.run_phase1`
2. Trained Branch B, Branch A, and Fusion with:
   - `python -m src.train_models --split-dir data/processed/phase1 --output-dir models/phase2`

## Data summary
- Combined rows: 366
- Balanced rows: 184
- Train rows: 128
- Validation rows: 19
- Test rows: 37

## Validation metrics (macro-F1)
- Branch B (stylistic): 0.7368
- Branch A (semantic baseline): 0.9468
- Fusion: 0.9468

## Test metrics (macro-F1)
- Branch B (stylistic): 0.8368
- Branch A (semantic baseline): 0.9189
- Fusion: 0.9729

## Generated artifacts
- `models/phase2/branch_b_style_model.joblib`
- `models/phase2/branch_a_semantic_model.joblib`
- `models/phase2/fusion_model.joblib`
- `models/phase2/training_metrics.json`

## Technical notes
1. Fixed label harmonization by forcing source labels in preprocessing (`real=0`, `fake=1`).
2. Fixed balancing behavior to preserve label column reliably across pandas behavior differences.
3. Fixed split logic to preserve engineered style feature columns in train/val/test outputs.

## Interpretation note for report
Current semantic branch uses TF-IDF baseline; this training run validates end-to-end branch + fusion plumbing and ablation protocol before RoBERTa upgrade.
