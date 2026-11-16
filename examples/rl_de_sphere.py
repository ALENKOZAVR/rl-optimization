from __future__ import annotations

import argparse
from pathlib import Path

from rl_optimization.algorithms.differential_evolution import DEConfig
from rl_optimization.algorithms.differential_evolution import DEDebugConfig
from rl_optimization.algorithms.differential_evolution import RLConfig
from rl_optimization.algorithms.differential_evolution import run_rl_differential_evolution
from rl_optimization.benchmarks.sphere import sphere
from rl_optimization.monitoring.history import save_history_csv
from rl_optimization.monitoring.history import save_result_json
from rl_optimization.monitoring.history import save_simple_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL-DE (Q-learning) on Sphere")
    parser.add_argument("--mode", choices=["normal", "debug"], default="normal")
    parser.add_argument("--output-dir", default="results_rl_de")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEConfig(
        num_variables=2,
        left_border=-5.12,
        right_border=5.12,
        pop_size=10,
        F=0.5,
        CR=0.9,
        iters=10,
        seed=42,
        objective_mode="population",
        minimization=True,
        elitism=True,
    )
    rl_config = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42, carry_learning=False)
    debug_config = DEDebugConfig(
        enabled=args.mode == "debug",
        generation_to_trace=1,
        individual_index_to_trace=0,
    )

    result = run_rl_differential_evolution(
        objective=sphere,
        config=config,
        rl_config=rl_config,
        debug_config=debug_config,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_result_json(result.to_dict(), out_dir / "result.json")
    save_history_csv(result.best_value_history, out_dir / "history.csv")
    save_simple_plot(result.best_value_history, out_dir / "best_value_plot.png", title="RL-DE on Sphere")

    print("=== RL-DE (Q-learning + thefittest DE) ===")
    print(f"thefittest version: {result.thefittest_version}")
    print(f"mode: {args.mode}")
    print(f"best solution: {result.best_solution}")
    print(f"best value (Sphere): {result.best_value}")
    print(f"function evaluations: {result.function_evaluations}")
    print(f"strategy history: {result.strategy_history}")
    print(f"final Q-table: {result.final_q_table}")
    print(f"RL transitions saved: {len(result.rl_history or [])}")
    if args.mode == "debug":
        print("debug trace:")
        print(result.debug_trace)
    print(f"saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
