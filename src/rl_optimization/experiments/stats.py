from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def compute_basic_stats(values: list[float], ddof: int = 1) -> dict[str, float]:
    if len(values) == 0:
        raise ValueError("values must not be empty")
    arr = np.asarray(values, dtype=np.float64)
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")
    if ddof == 1 and len(arr) < 2:
        raise ValueError("ddof=1 requires at least two values")
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=ddof)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def rank_algorithms(
    results: dict[str, list[float]],
    minimization: bool = True,
) -> dict[str, Any]:
    if len(results) == 0:
        raise ValueError("results must not be empty")
    run_counts = {name: len(vals) for name, vals in results.items()}
    if len(set(run_counts.values())) != 1:
        raise ValueError("all algorithms must have the same number of runs")
    n_runs = next(iter(run_counts.values()))
    if n_runs == 0:
        raise ValueError("each algorithm must have at least one run")

    ranks_by_algorithm: dict[str, list[float]] = {name: [] for name in results}
    names = list(results.keys())
    for run_idx in range(n_runs):
        row = [(name, float(results[name][run_idx])) for name in names]
        row.sort(key=lambda x: x[1], reverse=not minimization)
        rank_values: dict[str, float] = {}
        position = 1
        i = 0
        while i < len(row):
            j = i
            while j + 1 < len(row) and row[j + 1][1] == row[i][1]:
                j += 1
            avg_rank = float((position + position + (j - i)) / 2.0)
            for k in range(i, j + 1):
                rank_values[row[k][0]] = avg_rank
            position += (j - i + 1)
            i = j + 1
        for name in names:
            ranks_by_algorithm[name].append(rank_values[name])

    mean_rank = {name: float(np.mean(vals)) for name, vals in ranks_by_algorithm.items()}
    return {"ranks_by_algorithm": ranks_by_algorithm, "mean_rank": mean_rank}


def compute_strategy_stage_frequencies(
    rl_histories: list[list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    return compute_strategy_stage_frequencies_complete(rl_histories, as_percent=False)


def compute_strategy_stage_frequencies_complete(
    rl_histories: list[list[dict[str, Any]]],
    as_percent: bool = False,
    states: tuple[str, str, str] = ("early", "middle", "late"),
    actions: tuple[str, str, str] = ("rand_1", "best_1", "current_to_best_1"),
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for history in rl_histories:
        for step in history:
            state = str(step["state"])
            action = str(step["action"])
            counts[state][action] += 1
            totals[state] += 1

    frequencies: dict[str, dict[str, float]] = {}
    scale = 100.0 if as_percent else 1.0
    for state in states:
        state_total = totals[state]
        frequencies[state] = {}
        for action in actions:
            count = counts[state][action]
            value = (count / state_total if state_total > 0 else 0.0) * scale
            frequencies[state][action] = float(value)
    return frequencies


def rank_algorithms_by_function_means(
    function_results: dict[str, dict[str, list[float]]],
    minimization: bool = True,
    ddof: int = 1,
) -> dict[str, Any]:
    if len(function_results) == 0:
        raise ValueError("function_results must not be empty")
    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")

    per_function_stats: dict[str, dict[str, dict[str, float]]] = {}
    per_function_ranks: dict[str, dict[str, float]] = {}
    mean_rank_acc: dict[str, list[float]] = defaultdict(list)

    for function_name, by_algorithm in function_results.items():
        if len(by_algorithm) == 0:
            raise ValueError(f"No algorithms provided for function '{function_name}'")
        stats_for_function: dict[str, dict[str, float]] = {}
        for algo_name, values in by_algorithm.items():
            stats_for_function[algo_name] = compute_basic_stats(values, ddof=ddof)
        per_function_stats[function_name] = stats_for_function

        row = [(algo, stats_for_function[algo]["mean"]) for algo in stats_for_function.keys()]
        row.sort(key=lambda x: x[1], reverse=not minimization)

        ranks: dict[str, float] = {}
        i = 0
        position = 1
        while i < len(row):
            j = i
            while j + 1 < len(row) and row[j + 1][1] == row[i][1]:
                j += 1
            avg_rank = float((position + position + (j - i)) / 2.0)
            for k in range(i, j + 1):
                ranks[row[k][0]] = avg_rank
                mean_rank_acc[row[k][0]].append(avg_rank)
            position += (j - i + 1)
            i = j + 1

        per_function_ranks[function_name] = ranks

    mean_rank = {algo: float(np.mean(vals)) for algo, vals in mean_rank_acc.items()}
    return {
        "per_function_stats": per_function_stats,
        "per_function_ranks": per_function_ranks,
        "mean_rank": mean_rank,
        "std_ddof": ddof,
        "tie_rule": "exact_equality_on_mean_values",
    }
