# Model Training Change 05: Branch A, Branch B, and Fusion Training Pipeline

## Why this change
After Phase 1 data contract and deterministic splits, the next critical step is to train the three experimental variants required by the architecture and ablation design.

## What was implemented
A new training module was added in `src/training.py` with:
1. Branch B stylistic model training using engineered style features.
2. Branch A semantic model training using TF-IDF text features as a practical semantic baseline.
3. Fusion model training using branch confidence scores as fusion inputs.
4. Test-set evaluation for style-only, semantic-only, and fusion outputs.
5. Artifact persistence for all models and metrics JSON.

## Report value
This operationalizes the ablation protocol: Branch B vs Branch A vs Fusion under identical train/val/test splits.

## Notes for thesis framing
- Current Branch A uses a TF-IDF semantic baseline to move quickly into experimental training.
- RoBERTa-based Branch A can replace this module in a later phase while keeping the same training/evaluation interfaces.

## Affected files
- `src/training.py`
- `src/train_models.py`
