from __future__ import annotations

import argparse
import json

try:
    from .training import run_training_from_phase1
except ImportError:
    from training import run_training_from_phase1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Branch A, Branch B, and Fusion models")
    parser.add_argument("--split-dir", default="data/processed/phase1", help="Directory containing train.csv/val.csv/test.csv")
    parser.add_argument("--output-dir", default="models/phase2", help="Directory for trained model artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training_from_phase1(split_dir=args.split_dir, model_output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
