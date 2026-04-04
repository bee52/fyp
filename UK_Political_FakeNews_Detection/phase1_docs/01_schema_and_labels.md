# Phase 1 Change 01: Schema and Label Standardization

## Why this change
Your two scrapers were writing labels in mixed formats (`REAL` string in one file and `1` integer in another), which breaks deterministic preprocessing and can silently corrupt class balancing.

## What was implemented
1. Added a canonical schema utility module in `src/schema.py`.
2. Standardized labels to integers (`0` = real, `1` = fake/satire).
3. Added record/dataframe normalization helpers to enforce column shape:
   - `title`
   - `text`
   - `source`
   - `date`
   - `label`

## Report value
This establishes a strict data contract at ingestion time, reducing label drift and making downstream train/val/test splits reproducible.

## Affected files
- `src/schema.py`
- `src/bulk_uk_scraper.py`
- `src/bulk_uk_satire_scraper.py`
