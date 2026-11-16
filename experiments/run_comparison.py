from __future__ import annotations

import argparse
import json

from rl_optimization.experiments.comparison import run_comparison_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one comparison experiment from YAML config")
    parser.add_argument("--config", default="experiments/configs/rl_de_comparison_check.yaml")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["de_rand_1", "de_best_1", "de_current_to_best_1", "random_de", "rl_de_q_learning"],
    )
    parser.add_argument(
        "--function",
        required=True,
        choices=["sphere", "ackley", "rastrigin", "rosenbrock", "schwefel"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", default="results_comparison")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = run_comparison_experiment(
        config_path=args.config,
        algorithm=args.algorithm,
        function_name=args.function,
        repeats=args.repeats,
        output_dir=args.output_dir,
        debug=args.debug,
    )
    print(json.dumps({"output_path": out["output_path"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
