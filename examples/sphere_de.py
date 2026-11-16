from __future__ import annotations

import argparse
from pathlib import Path

from rl_optimization.algorithms.differential_evolution import DEConfig
from rl_optimization.algorithms.differential_evolution import DEDebugConfig
from rl_optimization.algorithms.differential_evolution import normalize_mutation_strategy
from rl_optimization.algorithms.differential_evolution import run_differential_evolution
from rl_optimization.benchmarks.sphere import sphere
from rl_optimization.monitoring.history import save_history_csv
from rl_optimization.monitoring.history import save_result_json
from rl_optimization.monitoring.history import save_simple_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DE on Sphere using thefittest")
    parser.add_argument("--mode", choices=["normal", "debug"], default="normal")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--strategy",
        default="rand/1",
        help="Mutation strategy: rand/1, best/1, current-to-best/1, random-de (also aliases with *_1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mutation = normalize_mutation_strategy(args.strategy)

    config = DEConfig(
        num_variables=2,
        left_border=-5.12,
        right_border=5.12,
        pop_size=10,
        F=0.5,
        CR=0.9,
        iters=10,
        seed=42,
        mutation=mutation,
        minimization=True,
        elitism=True,
    )
    debug_config = DEDebugConfig(
        enabled=args.mode == "debug",
        generation_to_trace=1,
        individual_index_to_trace=0,
    )

    result = run_differential_evolution(sphere, config=config, debug_config=debug_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_result_json(result.to_dict(), out_dir / "result.json")
    save_history_csv(result.best_value_history, out_dir / "history.csv")
    save_simple_plot(result.best_value_history, out_dir / "best_value_plot.png")

    print("=== Differential Evolution (thefittest) ===")
    print(f"thefittest version: {result.thefittest_version}")
    print(f"mode: {args.mode}")
    print(f"strategy: {mutation}")
    print(f"best solution: {result.best_solution}")
    print(f"best value (Sphere): {result.best_value}")
    print(f"function evaluations: {result.function_evaluations}")
    print(f"best value history: {result.best_value_history}")
    print(f"strategy history: {result.strategy_history}")
    if args.mode == "debug":
        print("debug trace (one individual, one generation transition):")
        print(result.debug_trace)
    print(f"saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
