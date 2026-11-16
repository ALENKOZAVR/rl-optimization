from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import csv

from rl_optimization.experiments.comparison import load_experiment_config
from rl_optimization.experiments.comparison import _parse_param
from rl_optimization.experiments.comparison import run_comparison_experiment
from rl_optimization.experiments.stats import compute_basic_stats
from rl_optimization.experiments.stats import compute_strategy_stage_frequencies_complete
from rl_optimization.experiments.stats import rank_algorithms_by_function_means


ALL_ALGORITHMS = [
    "de_rand_1",
    "de_best_1",
    "de_current_to_best_1",
    "random_de",
    "rl_de_q_learning",
]
ALL_FUNCTIONS = ["sphere", "ackley", "rastrigin", "rosenbrock", "schwefel"]


def _canonical_algorithm_name(algorithm: str) -> str:
    mapping = {
        "de_rand_1": "rand_1",
        "de_best_1": "best_1",
        "de_current_to_best_1": "current_to_best_1",
        "random_de": "random_de",
        "rl_de_q_learning": "rl_de_q_learning",
        "rl_de_q_learning_stagnation": "rl_de_q_learning_stagnation",
    }
    return mapping.get(algorithm, algorithm)


def _run_file_path(
    runs_dir: Path,
    function_name: str,
    algorithm: str,
    run_index: int,
    optimizer_seed: int,
) -> Path:
    algo_name = _canonical_algorithm_name(algorithm)
    return runs_dir / function_name / algo_name / f"run_{run_index:03d}_seed_{optimizer_seed}.json"


def _is_complete_run_file(
    run_file: Path,
    *,
    expected_algorithm: str,
    expected_function: str,
    expected_run_index: int,
    expected_seed: int,
    expected_fes: int,
    expected_dimension: int,
    expected_bounds: tuple[float, float],
    expected_generations: int,
) -> bool:
    if not run_file.exists():
        return False
    try:
        payload = json.loads(run_file.read_text(encoding="utf-8"))
        run = payload["run"]
        result = run["result"]
        config = result["config"]

        if str(payload["metadata"]["algorithm"]) != expected_algorithm:
            return False
        if str(payload["metadata"]["function_name"]) != expected_function:
            return False
        if int(run["run_index"]) != expected_run_index:
            return False
        if int(run["optimizer_seed"]) != expected_seed:
            return False
        if int(result["function_evaluations"]) != expected_fes:
            return False
        if int(config["num_variables"]) != expected_dimension:
            return False
        left, right = expected_bounds
        if float(config["left_border"]) != left or float(config["right_border"]) != right:
            return False

        best_value = float(result["best_value"])
        if not math.isfinite(best_value):
            return False
        best_solution = result["best_solution"]
        if not isinstance(best_solution, list) or len(best_solution) != expected_dimension:
            return False
        for x in best_solution:
            xv = float(x)
            if not math.isfinite(xv) or xv < left or xv > right:
                return False

        best_hist = result.get("best_value_history")
        if not isinstance(best_hist, list) or len(best_hist) != expected_generations:
            return False

        algo = expected_algorithm
        strategy_history = result.get("strategy_history")
        if algo in {"random_de", "rl_de_q_learning", "rl_de_q_learning_stagnation"}:
            if not isinstance(strategy_history, list) or len(strategy_history) != expected_generations:
                return False

        if algo in {"rl_de_q_learning", "rl_de_q_learning_stagnation"}:
            rl_history = result.get("rl_history")
            if not isinstance(rl_history, list):
                return False
            if len(rl_history) != max(expected_generations - 1, 0):
                return False
            for step in rl_history:
                reward = float(step["reward"])
                if reward < 0.0 or reward > 1.0:
                    return False

        return True
    except Exception:
        return False


def run_batch_comparison(
    config_path: str | Path,
    repeats: int | None = None,
    output_dir: str | Path = "results_comparison_batch",
    debug: bool = False,
    algorithms: list[str] | None = None,
    functions: list[str] | None = None,
    resume: bool = False,
    batch_name: str | None = None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    repeats_to_use = (
        int(config["experiment"]["independent_runs"]["value"]) if repeats is None else int(repeats)
    )
    if repeats_to_use < 1:
        raise ValueError("repeats must be >= 1")

    algorithms_to_use = algorithms or ALL_ALGORITHMS
    functions_to_use = functions or ALL_FUNCTIONS

    if batch_name is not None and batch_name.strip() == "":
        raise ValueError("batch_name must not be empty")
    batch_id = batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S_%f")
    batch_dir = Path(output_dir) / batch_id
    if resume and not batch_dir.exists():
        raise ValueError(f"Cannot resume: batch directory does not exist: {batch_dir}")
    runs_dir = batch_dir / "runs"
    summary_dir = batch_dir / "summary"
    payloads_dir = batch_dir / "payloads"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    payloads_dir.mkdir(parents=True, exist_ok=True)

    all_payloads: list[dict[str, Any]] = []
    per_function_results: dict[str, dict[str, list[float]]] = {
        fn: {} for fn in functions_to_use
    }
    rl_histories_all: list[list[dict[str, Any]]] = []
    rl_histories_by_function: dict[str, list[list[dict[str, Any]]]] = {
        fn: [] for fn in functions_to_use
    }
    task_cfg = config["task"]
    de_cfg = config["de"]
    exp_cfg = config["experiment"]
    bounds_by_function = _parse_param(task_cfg, "bounds_by_function").value
    dimension = int(_parse_param(task_cfg, "dimension").value)
    max_fes = int(_parse_param(de_cfg, "max_fes").value)
    generations = int(_parse_param(de_cfg, "generations").value)
    base_seed = int(_parse_param(exp_cfg, "base_seed").value)
    state_order: list[str] = []

    skipped_completed_runs = 0
    executed_runs = 0

    for function_name in functions_to_use:
        for algorithm in algorithms_to_use:
            algo_name = _canonical_algorithm_name(algorithm)
            best_values: list[float] = []
            left, right = bounds_by_function[function_name]
            bounds = (float(left), float(right))

            for run_index in range(repeats_to_use):
                optimizer_seed = base_seed + run_index
                run_file = _run_file_path(
                    runs_dir=runs_dir,
                    function_name=function_name,
                    algorithm=algorithm,
                    run_index=run_index,
                    optimizer_seed=optimizer_seed,
                )
                if resume and _is_complete_run_file(
                    run_file,
                    expected_algorithm=algo_name,
                    expected_function=function_name,
                    expected_run_index=run_index,
                    expected_seed=optimizer_seed,
                    expected_fes=max_fes,
                    expected_dimension=dimension,
                    expected_bounds=bounds,
                    expected_generations=generations,
                ):
                    skipped_completed_runs += 1
                else:
                    result = run_comparison_experiment(
                        config_path=config_path,
                        algorithm=algorithm,
                        function_name=function_name,
                        repeats=1,
                        output_dir=payloads_dir,
                        debug=debug,
                        run_offset=run_index,
                        enforce_reset_q_each_run_for_comparison=True,
                    )
                    payload = result["payload"]
                    all_payloads.append(payload)
                    run = payload["runs"][0]
                    run_file.parent.mkdir(parents=True, exist_ok=True)
                    run_payload = {
                        "metadata": payload["metadata"],
                        "config": payload["config"],
                        "run": run,
                    }
                    with run_file.open("w", encoding="utf-8") as f:
                        json.dump(run_payload, f, ensure_ascii=False, indent=2)
                    executed_runs += 1

                loaded = json.loads(run_file.read_text(encoding="utf-8"))
                loaded_run = loaded["run"]
                run_result = loaded_run["result"]
                best_values.append(float(run_result["best_value"]))
                rl_history = run_result.get("rl_history")
                if rl_history is not None:
                    rl_histories_all.append(rl_history)
                    rl_histories_by_function[function_name].append(rl_history)
                    for step in rl_history:
                        state = str(step["state"])
                        if state not in state_order:
                            state_order.append(state)

            per_function_results[function_name][algo_name] = best_values

    ddof_requested = int(config["experiment"]["std_ddof"]["value"])
    ddof_used = 0 if (ddof_requested == 1 and repeats_to_use < 2) else ddof_requested
    ranking = rank_algorithms_by_function_means(
        function_results=per_function_results,
        minimization=bool(config["task"]["minimization"]["value"]),
        ddof=ddof_used,
    )

    summary_rows: list[dict[str, Any]] = []
    for function_name, by_algo in per_function_results.items():
        for algo_name, values in by_algo.items():
            st = compute_basic_stats(values, ddof=ddof_used)
            rank_val = ranking["per_function_ranks"][function_name][algo_name]
            summary_rows.append(
                {
                    "function": function_name,
                    "algorithm": algo_name,
                    "mean": st["mean"],
                    "std": st["std"],
                    "mean_plus_minus_std": f"{st['mean']:.10g} ± {st['std']:.10g}",
                    "min": st["min"],
                    "max": st["max"],
                    "rank": rank_val,
                }
            )

    states_for_frequency = tuple(state_order) if state_order else ("early", "middle", "late")
    rl_stage_freq_ratio = compute_strategy_stage_frequencies_complete(
        rl_histories_all, as_percent=False, states=states_for_frequency
    )
    rl_stage_freq_percent = compute_strategy_stage_frequencies_complete(
        rl_histories_all, as_percent=True, states=states_for_frequency
    )

    rl_stage_freq_by_function: dict[str, dict[str, dict[str, float]]] = {}
    for function_name, histories in rl_histories_by_function.items():
        rl_stage_freq_by_function[function_name] = compute_strategy_stage_frequencies_complete(
            histories, as_percent=False, states=states_for_frequency
        )

    summary_json = {
        "batch_id": batch_id,
        "config_path": str(Path(config_path).resolve()),
        "repeats_used": repeats_to_use,
        "std_ddof_requested": ddof_requested,
        "std_ddof_used": ddof_used,
        "algorithms": algorithms_to_use,
        "functions": functions_to_use,
        "summary_rows": summary_rows,
        "mean_rank": ranking["mean_rank"],
        "ranking_details": ranking,
        "rl_stage_frequencies_ratio": rl_stage_freq_ratio,
        "rl_stage_frequencies_percent": rl_stage_freq_percent,
        "rl_stage_frequencies_by_function_ratio": rl_stage_freq_by_function,
        "run_files_root": str(runs_dir.resolve()),
        "executed_runs": executed_runs,
        "skipped_completed_runs": skipped_completed_runs,
    }
    summary_json_path = summary_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    summary_csv_path = summary_dir / "summary_table.csv"
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "function",
                "algorithm",
                "mean",
                "std",
                "mean_plus_minus_std",
                "min",
                "max",
                "rank",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    mean_rank_csv = summary_dir / "mean_rank.csv"
    with mean_rank_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "mean_rank"])
        writer.writeheader()
        for algo_name, rank_val in sorted(ranking["mean_rank"].items(), key=lambda x: x[1]):
            writer.writerow({"algorithm": algo_name, "mean_rank": rank_val})

    rl_freq_csv = summary_dir / "rl_stage_frequencies_ratio.csv"
    with rl_freq_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state", "action", "frequency_ratio", "frequency_percent"])
        writer.writeheader()
        for state, action_map in rl_stage_freq_ratio.items():
            for action, ratio in action_map.items():
                writer.writerow(
                    {
                        "state": state,
                        "action": action,
                        "frequency_ratio": ratio,
                        "frequency_percent": rl_stage_freq_percent[state][action],
                    }
                )

    return {
        "batch_dir": str(batch_dir),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
    }
