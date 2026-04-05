# Refactored Training Architecture

## New Module Separation

### 1. Scikit-learn Stack: `src/training.py`
- **Branch B (Stylistic)**: RandomForestClassifier (matching notebook approach)
- **Branch A (Semantic)**: TF-IDF + Logistic Regression (fast baseline)
- Entrypoint: `python -m src.train_models` (existing command unchanged)

### 2. RoBERTa Deep Learning: `src/training_roberta.py`
- **Branch A (Semantic)**: DistilRoBERTa embeddings + Logistic head
- Uses `sentence-transformers` for efficient transformer encoding
- Supports device selection (cpu, cuda, mps)
- Entrypoint: `python -m src.train_models_roberta`

### 3. Fusion Layer: `src/fusion.py`
- `train_sklearn_fusion()`: For scikit Branch B + semantic (TF-IDF or RoBERTa)
- `evaluate_sklearn_fusion_on_test()`: Test evaluation for scikit branches
- `train_roberta_fusion()`: Fusion of stylistic scores + RoBERTa embeddings  
- `evaluate_roberta_fusion_on_test()`: Test evaluation with RoBERTa

## Training Commands

### Option 1: Scikit-Only (Fast)
```bash
python -m src.train_models \
  --split-dir data/processed/phase1 \
  --output-dir models/phase2
```
Output: `models/phase2/training_metrics.json`

### Option 2: RoBERTa-Based (Deep Learning)
```bash
python -m src.train_models_roberta \
  --split-dir data/processed/phase1 \
  --output-dir models/phase2_roberta \
  --roberta-model distilroberta-base \
  --device cpu
```
Output: `models/phase2_roberta/training_metrics_roberta.json`

## Architecture Benefits

1. **Clear Separation**: Scikit code isolated from transformer code
2. **Easy Comparison**: Run both variants on same splits to measure RoBERTa gains
3. **Modularity**: Fusion functions work with either branch combination
4. **Reproducibility**: Each run logs config, model names, device, and metrics

## Dependencies

- Scikit-only: `pandas`, `scikit-learn`, `numpy` (already installed)
- RoBERTa: Also requires `sentence-transformers>=3.0.0`, `torch>=2.4.0` (see `requirements.txt`)
