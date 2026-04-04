# Phase 1 Change 03: Reusable Preprocessing Module

## Why this change
Core preprocessing logic existed mainly in notebooks, making the pipeline difficult to test, reuse, or run headlessly.

## What was implemented
Created `src/preprocessing.py` with reusable functions for:
1. loading and harmonizing real/fake CSV data,
2. dataset balancing with fixed random seed,
3. text cleanup,
4. stylistic feature extraction,
5. deterministic stratified train/val/test split,
6. writing split artifacts and metadata.

## Report value
This converts notebook logic into testable production-style code and sets up deterministic Phase 1 data outputs for later Branch A/Branch B/GLU training.

## Affected files
- `src/preprocessing.py`
