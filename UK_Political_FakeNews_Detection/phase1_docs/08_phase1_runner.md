# Phase 1 Change 08: Automated Data Preparation Runner

## Why this change
To move into model training reliably, train/val/test artifacts must be generated with one reproducible command.

## What was implemented
Added `src/run_phase1.py` to execute Phase 1 pipeline logic from terminal.

Behavior:
1. Uses explicit `--real-csv` and `--fake-csv` if provided.
2. Otherwise auto-selects latest files in `data/raw/` by prefixes:
   - `uk_politics_`
   - `uk_fake_satire_`
3. Calls `UKFakeNewsPipeline.run_phase1_data_preparation`.
4. Prints artifact paths and row counts as JSON.

## Command
```bash
python -m src.run_phase1
```

## Report value
This creates deterministic split artifacts required for fair ablation training.

## Affected files
- `src/run_phase1.py`
