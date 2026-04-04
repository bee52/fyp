# Phase 1 Change 04: Phase-1 Pipeline Entry Path

## Why this change
The existing pipeline class focused on placeholder inference for UI and did not expose a reproducible data-preparation command path.

## What was implemented
Added a dedicated Phase 1 execution path in the pipeline class (`run_phase1_data_preparation`) for ingestion contract + preprocessing + split artifacts, separate from UI prediction placeholders.

## Report value
This creates a clear separation between:
- data engineering lifecycle (Phase 1), and
- inference UI lifecycle (demo app).

## Affected files
- `src/pipeline.py`
- `src/app.py` (kept as UI shell)
