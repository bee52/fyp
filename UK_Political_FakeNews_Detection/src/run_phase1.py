from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .pipeline import UKFakeNewsPipeline
except ImportError:
    from pipeline import UKFakeNewsPipeline


def _latest_file(folder: Path, prefix: str) -> Path:
    candidates = sorted(folder.glob(f"{prefix}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No files found with prefix '{prefix}' in {folder}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 data preparation")
    parser.add_argument("--real-csv", default=None, help="Path to real-news CSV")
    parser.add_argument("--fake-csv", default=None, help="Path to fake/satire CSV")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = UKFakeNewsPipeline(config_path=args.config)

    raw_dir = Path("data/raw")
    real_csv = Path(args.real_csv) if args.real_csv else _latest_file(raw_dir, "uk_politics_")
    fake_csv = Path(args.fake_csv) if args.fake_csv else _latest_file(raw_dir, "uk_fake_satire_")

    result = pipeline.run_phase1_data_preparation(real_csv=str(real_csv), fake_csv=str(fake_csv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
