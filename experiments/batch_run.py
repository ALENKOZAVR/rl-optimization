from __future__ import annotations

import argparse
import json

from rl_optimization.experiments.batch import run_batch_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run DE/RL-DE comparison")
    parser.add_argument("--config", default="experiments/configs/rl_de_comparison_check.yaml")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--output-dir", default="results_comparison_batch")
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_batch_comparison(
        config_path=args.config,
        repeats=args.repeats,
        output_dir=args.output_dir,
        batch_name=args.batch_name,
        resume=args.resume,
        debug=args.debug,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
