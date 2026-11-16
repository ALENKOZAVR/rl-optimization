from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rl_optimization.experiments.stats import compute_basic_stats
from rl_optimization.experiments.stats import compute_strategy_stage_frequencies_complete
from rl_optimization.experiments.stats import rank_algorithms_by_function_means


def _load_run_files(runs_root: Path) -> list[dict[str, Any]]:
    run_files = sorted(runs_root.rglob("run_*.json"))
    records: list[dict[str, Any]] = []
    for path in run_files:
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def analyze_batch(batch_dir: str | Path) -> dict[str, Any]:
    batch_path = Path(batch_dir)
    runs_root = batch_path / "runs"
    summary_root = batch_path / "summary"
    analysis_root = batch_path / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    records = _load_run_files(runs_root)
    if not records:
        raise ValueError(f"No run files found under {runs_root}")

    by_function_algo: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rec in records:
        md = rec["metadata"]
        run = rec["run"]
        function_name = str(md["function_name"])
        algo = str(md["algorithm"])
        by_function_algo.setdefault(function_name, {}).setdefault(algo, []).append(rec)

    first_cfg = records[0]["config"]
    pop_size = int(first_cfg["de"]["population_size"]["value"])
    generations = int(first_cfg["de"]["generations"]["value"])
    max_fes = int(first_cfg["de"]["max_fes"]["value"])
    dimension = int(first_cfg["task"]["dimension"]["value"])
    ddof = int(first_cfg["experiment"]["std_ddof"]["value"])
    minimization = bool(first_cfg["task"]["minimization"]["value"])
    repeats = int(first_cfg["experiment"]["independent_runs"]["value"])
    expected_rl_steps = max(generations - 1, 0)

    integrity = {
        "total_run_files": len(records),
        "all_runs_completed": True,
        "all_budgets_match": True,
        "all_best_values_finite": True,
        "all_solutions_within_bounds": True,
        "all_histories_valid": True,
        "all_rl_steps_match_expected": True,
        "all_rl_rewards_in_range": True,
        "all_seeds_present": True,
        "no_duplicate_keys": True,
        "errors": [],
    }

    seen_keys: set[tuple[str, str, int, int]] = set()
    for rec in records:
        md = rec["metadata"]
        run = rec["run"]
        result = run["result"]
        cfg = result["config"]
        function_name = str(md["function_name"])
        algo = str(md["algorithm"])
        run_index = int(run["run_index"])
        optimizer_seed = int(run["optimizer_seed"])
        key = (function_name, algo, run_index, optimizer_seed)
        if key in seen_keys:
            integrity["no_duplicate_keys"] = False
            integrity["errors"].append(f"duplicate run key: {key}")
        seen_keys.add(key)

        if int(result["function_evaluations"]) != max_fes:
            integrity["all_budgets_match"] = False
            integrity["errors"].append(f"budget mismatch: {key}")
        if not _is_finite_number(result["best_value"]):
            integrity["all_best_values_finite"] = False
            integrity["errors"].append(f"non-finite best value: {key}")

        bounds = md["actual_bounds"]
        left, right = float(bounds[0]), float(bounds[1])
        best_solution = result["best_solution"]
        if len(best_solution) != dimension:
            integrity["all_solutions_within_bounds"] = False
            integrity["errors"].append(f"solution dimension mismatch: {key}")
        else:
            for x in best_solution:
                xv = float(x)
                if not math.isfinite(xv) or xv < left or xv > right:
                    integrity["all_solutions_within_bounds"] = False
                    integrity["errors"].append(f"out-of-bounds solution: {key}")
                    break

        hist = result.get("best_value_history", [])
        if not isinstance(hist, list) or len(hist) != generations:
            integrity["all_histories_valid"] = False
            integrity["errors"].append(f"history length mismatch: {key}")

        if run.get("optimizer_seed") is None:
            integrity["all_seeds_present"] = False
            integrity["errors"].append(f"missing optimizer seed: {key}")
        if algo == "random_de" and run.get("strategy_selector_seed") is None:
            integrity["all_seeds_present"] = False
            integrity["errors"].append(f"missing random_de strategy seed: {key}")
        if algo == "rl_de_q_learning" and run.get("rl_seed") is None:
            integrity["all_seeds_present"] = False
            integrity["errors"].append(f"missing rl seed: {key}")

        if algo in {"random_de", "rl_de_q_learning"}:
            sh = result.get("strategy_history")
            if not isinstance(sh, list) or len(sh) != generations:
                integrity["all_histories_valid"] = False
                integrity["errors"].append(f"strategy history mismatch: {key}")

        if algo == "rl_de_q_learning":
            rl_history = result.get("rl_history")
            if not isinstance(rl_history, list) or len(rl_history) != expected_rl_steps:
                integrity["all_rl_steps_match_expected"] = False
                integrity["errors"].append(f"rl steps mismatch: {key}")
            else:
                for step in rl_history:
                    reward = float(step["reward"])
                    if reward < 0.0 or reward > 1.0:
                        integrity["all_rl_rewards_in_range"] = False
                        integrity["errors"].append(f"rl reward out of range: {key}")
                        break

        if int(cfg["num_variables"]) != dimension or int(cfg["iters"]) != generations:
            integrity["all_runs_completed"] = False
            integrity["errors"].append(f"config mismatch: {key}")

    required_flags = [
        "all_budgets_match",
        "all_best_values_finite",
        "all_solutions_within_bounds",
        "all_histories_valid",
        "all_rl_steps_match_expected",
        "all_rl_rewards_in_range",
        "all_seeds_present",
        "no_duplicate_keys",
    ]
    integrity["all_runs_completed"] = all(bool(integrity[f]) for f in required_flags)

    # Table 1 + rank data
    function_results: dict[str, dict[str, list[float]]] = {}
    summary_rows: list[dict[str, Any]] = []
    successful_runs_rows: list[dict[str, Any]] = []
    for function_name, by_algo in sorted(by_function_algo.items()):
        function_results[function_name] = {}
        for algo, recs in sorted(by_algo.items()):
            values = [float(r["run"]["result"]["best_value"]) for r in recs]
            function_results[function_name][algo] = values
            stats = compute_basic_stats(values, ddof=ddof)
            summary_rows.append(
                {
                    "function": function_name,
                    "algorithm": algo,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "mean_plus_minus_std": f"{stats['mean']:.10g} ± {stats['std']:.10g}",
                    "min": stats["min"],
                    "max": stats["max"],
                    "successful_runs": len(values),
                }
            )
            successful_runs_rows.append(
                {
                    "function": function_name,
                    "algorithm": algo,
                    "successful_runs": len(values),
                    "expected_runs": repeats,
                }
            )

    ranking = rank_algorithms_by_function_means(
        function_results=function_results,
        minimization=minimization,
        ddof=ddof,
    )

    table1_csv = analysis_root / "table1_mean_std.csv"
    with table1_csv.open("w", encoding="utf-8", newline="") as f:
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
                "successful_runs",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    table2_rows: list[dict[str, Any]] = []
    for fn_name, ranks in ranking["per_function_ranks"].items():
        for algo, rank in ranks.items():
            table2_rows.append({"function": fn_name, "algorithm": algo, "rank": rank})
    for algo, mr in ranking["mean_rank"].items():
        table2_rows.append({"function": "MEAN", "algorithm": algo, "rank": mr})

    table2_csv = analysis_root / "table2_ranks.csv"
    with table2_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["function", "algorithm", "rank"])
        writer.writeheader()
        for row in table2_rows:
            writer.writerow(row)

    successful_runs_csv = analysis_root / "successful_runs.csv"
    with successful_runs_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["function", "algorithm", "successful_runs", "expected_runs"])
        writer.writeheader()
        for row in successful_runs_rows:
            writer.writerow(row)

    # Figure 1: convergence (mean +- std across runs), x = actual FEs.
    fig1_path = analysis_root / "figure1_convergence.png"
    functions = sorted(by_function_algo.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    for i, fn_name in enumerate(functions):
        ax = axes_flat[i]
        for algo, recs in sorted(by_function_algo[fn_name].items()):
            histories = np.asarray(
                [r["run"]["result"]["best_value_history"] for r in recs],
                dtype=np.float64,
            )
            mean_hist = np.mean(histories, axis=0)
            std_hist = np.std(histories, axis=0, ddof=ddof)
            x = np.arange(1, generations + 1, dtype=np.int64) * pop_size
            ax.plot(x, mean_hist, label=algo)
            ax.fill_between(x, mean_hist - std_hist, mean_hist + std_hist, alpha=0.15)
        ax.set_title(fn_name)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Best value")
        ax.grid(True, alpha=0.25)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig1_path, dpi=160)
    plt.close(fig)

    # RL analysis
    rl_records = [r for r in records if str(r["metadata"]["algorithm"]) == "rl_de_q_learning"]
    rl_histories = [r["run"]["result"]["rl_history"] for r in rl_records]
    rl_stage_freq_ratio = compute_strategy_stage_frequencies_complete(rl_histories, as_percent=False)
    rl_stage_freq_percent = compute_strategy_stage_frequencies_complete(rl_histories, as_percent=True)

    stage_rewards: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    stage_freq_by_function_ratio: dict[str, dict[str, dict[str, float]]] = {}
    for fn_name in functions:
        fn_histories = [
            r["run"]["result"]["rl_history"]
            for r in rl_records
            if str(r["metadata"]["function_name"]) == fn_name
        ]
        stage_freq_by_function_ratio[fn_name] = compute_strategy_stage_frequencies_complete(
            fn_histories, as_percent=False
        )
        for hist in fn_histories:
            for step in hist:
                stage_rewards[str(step["state"])].append(float(step["reward"]))

    mean_reward_by_stage = {
        stage: float(np.mean(vals)) if vals else 0.0 for stage, vals in stage_rewards.items()
    }

    final_q_tables = []
    for r in rl_records:
        hist = r["run"]["result"]["rl_history"]
        if hist:
            final_q_tables.append(hist[-1]["q_table_after_update"])

    # Figure 2: heatmap by function/stage/action.
    fig2_path = analysis_root / "figure2_rl_strategy_heatmap.png"
    actions = ["rand_1", "best_1", "current_to_best_1"]
    stages = ["early", "middle", "late"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for idx, stage in enumerate(stages):
        data = np.zeros((len(functions), len(actions)), dtype=np.float64)
        for i, fn_name in enumerate(functions):
            freq = stage_freq_by_function_ratio[fn_name][stage]
            for j, action in enumerate(actions):
                data[i, j] = float(freq[action]) * 100.0
        ax = axes[idx]
        im = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=100.0)
        ax.set_xticks(np.arange(len(actions)))
        ax.set_xticklabels(actions, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(functions)))
        ax.set_yticklabels(functions)
        ax.set_title(f"Stage: {stage}")
        for i in range(len(functions)):
            for j in range(len(actions)):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Selection frequency (%)")
    fig.savefig(fig2_path, dpi=160)
    plt.close(fig)

    analysis = {
        "batch_dir": str(batch_path.resolve()),
        "runs_root": str(runs_root.resolve()),
        "summary_root": str(summary_root.resolve()),
        "integrity": integrity,
        "table1_csv": str(table1_csv.resolve()),
        "table2_csv": str(table2_csv.resolve()),
        "successful_runs_csv": str(successful_runs_csv.resolve()),
        "figure1_convergence": str(fig1_path.resolve()),
        "figure2_rl_strategy_heatmap": str(fig2_path.resolve()),
        "ranking": ranking,
        "rl_stage_frequencies_ratio": rl_stage_freq_ratio,
        "rl_stage_frequencies_percent": rl_stage_freq_percent,
        "rl_stage_frequencies_by_function_ratio": stage_freq_by_function_ratio,
        "rl_mean_reward_by_stage": mean_reward_by_stage,
        "rl_final_q_tables": final_q_tables,
    }
    out_json = analysis_root / "analysis_report.json"
    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze batch experiment outputs")
    parser.add_argument("--batch-dir", required=True, help="Path to batch directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_batch(args.batch_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
