# Phase 1 Change 02: Central Configuration Layer

## Why this change
Hardcoded paths and scrape limits made experiments difficult to reproduce and compare across machines.

## What was implemented
1. Added root-level `config.yaml`.
2. Added `src/config.py` with:
   - project-root aware path resolution,
   - deep-merge config loading,
   - directory creation helper.
3. Introduced configuration sections for:
   - data paths,
   - split policy,
   - scraper behavior,
   - run mode and random seed.

## Report value
This enables controlled, documented experiment settings and removes environment-specific hardcoding.

## Affected files
- `config.yaml`
- `src/config.py`
- `src/bulk_uk_scraper.py`
- `src/bulk_uk_satire_scraper.py`
