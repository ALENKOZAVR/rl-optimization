from __future__ import annotations

import argparse
import json

from rl_optimization.experiments.stagnation_diagnostics import run_stagnation_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL-DE v2 stagnation diagnostics")
    parser.add_argument(
        "--v2-batch-dir",
        default="results/rl_de_v2_stagnation_10d_30runs",
        help="Directory containing v2 batch run outputs",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rl_de_v2_stagnation_diagnostics",
        help="Directory where diagnostics artifacts are saved",
    )
    parser.add_argument(
        "--stagnation-threshold",
        type=int,
        default=5,
        help="Stagnation threshold used in saved v2 runs",
    )
    parser.add_argument(
        "--relative-thresholds",
        nargs="+",
        type=float,
        default=[1e-3, 1e-4, 1e-5],
        help="Counterfactual relative-improvement thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_stagnation_diagnostics(
        v2_batch_dir=args.v2_batch_dir,
        output_dir=args.output_dir,
        stagnation_threshold=args.stagnation_threshold,
        relative_thresholds=tuple(args.relative_thresholds),
    )
    print(
        json.dumps(
            {
                "records_found": report["data_checks"]["records_found"],
                "runs_per_function": report["data_checks"]["runs_per_function"],
                "output_report": report["artifacts"]["improvement_stats_csv"].replace(
                    "improvement_stats_by_function.csv", "diagnostics_report.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
