from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FUNCTIONS: tuple[str, ...] = ("sphere", "ackley", "rastrigin", "rosenbrock", "schwefel")
STATE6: tuple[str, ...] = (
    "early_improving",
    "early_stagnating",
    "middle_improving",
    "middle_stagnating",
    "late_improving",
    "late_stagnating",
)
ACTIONS: tuple[str, ...] = ("rand_1", "best_1", "current_to_best_1")
STAGES: tuple[str, ...] = ("early", "middle", "late")


def _safe_float(x: Any) -> float:
    return float(x)


def relative_improvement(
    best_before: float,
    best_after: float,
    *,
    function_scale: float,
) -> float:
    if function_scale <= 0.0 or not math.isfinite(function_scale):
        raise ValueError("function_scale must be finite and > 0")
    abs_improvement = best_before - best_after
    denom = max(abs(best_before), function_scale)
    return abs_improvement / denom


def recompute_stagnation_from_best_history(
    best_value_history: list[float],
    *,
    minimization: bool,
    stagnation_threshold: int,
) -> dict[str, Any]:
    if stagnation_threshold <= 0:
        raise ValueError("stagnation_threshold must be > 0")
    if len(best_value_history) < 2:
        raise ValueError("best_value_history must contain at least 2 values")

    counters_before: list[int] = []
    counters_after: list[int] = []
    flags_before: list[str] = []
    flags_after: list[str] = []
    strict_improvements: list[bool] = []

    counter = 0
    for idx in range(1, len(best_value_history)):
        before = float(best_value_history[idx - 1])
        after = float(best_value_history[idx])
        counters_before.append(counter)
        flags_before.append("stagnating" if counter >= stagnation_threshold else "improving")

        improved = (after < before) if minimization else (after > before)
        strict_improvements.append(bool(improved))
        counter = 0 if improved else (counter + 1)

        counters_after.append(counter)
        flags_after.append("stagnating" if counter >= stagnation_threshold else "improving")

    return {
        "counters_before": counters_before,
        "counters_after": counters_after,
        "flags_before": flags_before,
        "flags_after": flags_after,
        "strict_improvements": strict_improvements,
    }


def compute_non_improvement_run_lengths(strict_improvements: list[bool]) -> list[int]:
    run_lengths: list[int] = []
    current = 0
    for improved in strict_improvements:
        if improved:
            if current > 0:
                run_lengths.append(current)
            current = 0
        else:
            current += 1
    if current > 0:
        run_lengths.append(current)
    return run_lengths


def _load_v2_records(runs_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(runs_root.rglob("run_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("metadata", {}).get("algorithm") == "rl_de_q_learning_stagnation":
            out.append(data)
    return out


def _function_scales(records: list[dict[str, Any]]) -> dict[str, float]:
    by_fn: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        fn = str(rec["metadata"]["function_name"])
        history = rec["run"]["result"]["best_value_history"]
        by_fn[fn].append(float(history[0]))
    scales: dict[str, float] = {}
    for fn, starts in by_fn.items():
        s = float(np.median(np.abs(np.asarray(starts, dtype=np.float64))))
        scales[fn] = s if s > 0.0 else 1.0
    return scales


def _stage_from_state(state: str) -> str:
    if "_" in state:
        return state.split("_", 1)[0]
    return state


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_stagnation_diagnostics(
    *,
    v2_batch_dir: str | Path,
    output_dir: str | Path,
    stagnation_threshold: int = 5,
    relative_thresholds: tuple[float, ...] = (1e-3, 1e-4, 1e-5),
) -> dict[str, Any]:
    v2_batch_dir = Path(v2_batch_dir)
    runs_root = v2_batch_dir / "runs"
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    records = _load_v2_records(runs_root)
    if len(records) == 0:
        raise ValueError(f"No RL-DE v2 run files found under {runs_root}")

    # ------------------------------------------------------------------
    # 1) Data checks
    # ------------------------------------------------------------------
    runs_per_function: dict[str, int] = {fn: 0 for fn in FUNCTIONS}
    best_history_lengths: dict[str, list[int]] = defaultdict(list)
    monotonic_violations = 0
    duplicate_keys = 0
    seen_keys: set[tuple[str, int]] = set()
    generations_detected: set[int] = set()
    pop_sizes_detected: set[int] = set()
    total_fes_detected: set[int] = set()

    for rec in records:
        fn = str(rec["metadata"]["function_name"])
        run = rec["run"]
        seed = int(run["optimizer_seed"])
        key = (fn, seed)
        if key in seen_keys:
            duplicate_keys += 1
        seen_keys.add(key)

        runs_per_function[fn] += 1
        res = run["result"]
        cfg = res["config"]
        history = [float(x) for x in res["best_value_history"]]
        best_history_lengths[fn].append(len(history))
        generations_detected.add(int(cfg["iters"]))
        pop_sizes_detected.add(int(cfg["pop_size"]))
        total_fes_detected.add(int(res["function_evaluations"]))

        arr = np.asarray(history, dtype=np.float64)
        if np.any(arr[1:] > arr[:-1] + 1e-15):
            monotonic_violations += 1

    function_scales = _function_scales(records)
    generations = max(generations_detected)
    pop_size = max(pop_sizes_detected)

    # ------------------------------------------------------------------
    # Per-transition extraction
    # ------------------------------------------------------------------
    transition_rows: list[dict[str, Any]] = []
    run_level_rows: list[dict[str, Any]] = []

    # state visit stats
    visits_by_state_total: dict[str, int] = {s: 0 for s in STATE6}
    rewards_by_state: dict[str, list[float]] = {s: [] for s in STATE6}
    actions_by_state: dict[str, dict[str, int]] = {
        s: {a: 0 for a in ACTIONS} for s in STATE6
    }
    state_visits_per_run: dict[str, list[int]] = {s: [] for s in STATE6}
    q_updates_per_state_action_per_run: dict[tuple[str, str], list[int]] = {
        (s, a): [] for s in STATE6 for a in ACTIONS
    }
    visits_heatmap_by_function_state: dict[str, dict[str, list[int]]] = {
        fn: {s: [] for s in STATE6} for fn in FUNCTIONS
    }

    # sensitivity aggregation
    sensitivity_rows: list[dict[str, Any]] = []

    for rec in records:
        md = rec["metadata"]
        fn = str(md["function_name"])
        run = rec["run"]
        seed = int(run["optimizer_seed"])
        res = run["result"]
        history = [float(x) for x in res["best_value_history"]]
        rl_history = res["rl_history"]
        recomputed = recompute_stagnation_from_best_history(
            history,
            minimization=True,
            stagnation_threshold=stagnation_threshold,
        )

        # per-run state visit counts
        per_run_state_count = {s: 0 for s in STATE6}
        per_run_sa_count = {(s, a): 0 for s in STATE6 for a in ACTIONS}

        positive_abs: list[float] = []
        positive_rel: list[float] = []
        strict_improvements = recomputed["strict_improvements"]

        for idx, step in enumerate(rl_history):
            transition = int(step["generation_transition"])
            best_before = history[transition - 1]
            best_after = history[transition]
            abs_impr = best_before - best_after
            rel_impr = relative_improvement(
                best_before,
                best_after,
                function_scale=function_scales[fn],
            )
            strict = bool(abs_impr > 0.0)
            if strict:
                positive_abs.append(abs_impr)
                positive_rel.append(rel_impr)

            state = str(step["state"])
            stage = str(step.get("stage", _stage_from_state(state)))
            action = str(step["action"])
            reward = float(step["reward"])

            row = {
                "function": fn,
                "optimizer_seed": seed,
                "run_index": int(run["run_index"]),
                "transition": transition,
                "fes_before_generation": pop_size * transition,
                "stage": stage,
                "state": state,
                "action": action,
                "reward": reward,
                "best_before": best_before,
                "best_after": best_after,
                "absolute_improvement": abs_impr,
                "relative_improvement": rel_impr,
                "strict_improvement": strict,
                "saved_stagnation_counter_before": int(step["stagnation_counter_before"]),
                "saved_stagnation_counter_after": int(step["stagnation_counter_after"]),
                "recomputed_stagnation_counter_before": int(recomputed["counters_before"][idx]),
                "recomputed_stagnation_counter_after": int(recomputed["counters_after"][idx]),
                "saved_flag_before": _stage_from_state(state).replace(stage, "").strip("_")
                if "_" in state
                else "improving",
                "recomputed_flag_before": str(recomputed["flags_before"][idx]),
            }
            transition_rows.append(row)

            if state in visits_by_state_total:
                visits_by_state_total[state] += 1
                rewards_by_state[state].append(reward)
                if action in actions_by_state[state]:
                    actions_by_state[state][action] += 1
                per_run_state_count[state] += 1
                per_run_sa_count[(state, action)] += 1

        for s in STATE6:
            state_visits_per_run[s].append(per_run_state_count[s])
            visits_heatmap_by_function_state[fn][s].append(per_run_state_count[s])
        for key, val in per_run_sa_count.items():
            q_updates_per_state_action_per_run[key].append(val)

        run_lengths = compute_non_improvement_run_lengths(strict_improvements)
        no_improve = int(np.count_nonzero(np.asarray(strict_improvements) == 0))
        improve = len(strict_improvements) - no_improve
        stag_generations = int(sum(1 for c in recomputed["counters_before"] if c >= stagnation_threshold))
        run_level_rows.append(
            {
                "function": fn,
                "optimizer_seed": seed,
                "run_index": int(run["run_index"]),
                "strict_improvement_generations": improve,
                "no_improvement_generations": no_improve,
                "stagnating_generations": stag_generations,
                "stagnating_share": stag_generations / len(strict_improvements),
                "max_non_improvement_streak": int(max(run_lengths) if run_lengths else 0),
                "mean_non_improvement_streak": float(np.mean(run_lengths) if run_lengths else 0.0),
                "times_reached_threshold": int(sum(1 for ln in run_lengths if ln >= stagnation_threshold)),
                "mean_positive_abs_improvement": float(np.mean(positive_abs) if positive_abs else 0.0),
                "median_positive_abs_improvement": float(np.median(positive_abs) if positive_abs else 0.0),
                "min_positive_abs_improvement": float(np.min(positive_abs) if positive_abs else 0.0),
                "max_positive_abs_improvement": float(np.max(positive_abs) if positive_abs else 0.0),
                "mean_positive_rel_improvement": float(np.mean(positive_rel) if positive_rel else 0.0),
                "median_positive_rel_improvement": float(np.median(positive_rel) if positive_rel else 0.0),
                "min_positive_rel_improvement": float(np.min(positive_rel) if positive_rel else 0.0),
                "max_positive_rel_improvement": float(np.max(positive_rel) if positive_rel else 0.0),
            }
        )

        # counterfactual sensitivity analysis on same trajectories
        best_arr = np.asarray(history, dtype=np.float64)
        rel_improvements = np.asarray(
            [
                relative_improvement(best_arr[i - 1], best_arr[i], function_scale=function_scales[fn])
                for i in range(1, len(best_arr))
            ],
            dtype=np.float64,
        )
        stage_seq = [str(step.get("stage", _stage_from_state(str(step["state"])))) for step in rl_history]
        for rel_thr in relative_thresholds:
            counter = 0
            flags_before: list[str] = []
            for r in rel_improvements:
                flags_before.append("stagnating" if counter >= stagnation_threshold else "improving")
                substantial = bool(r > rel_thr)
                counter = 0 if substantial else (counter + 1)

            i2s = 0
            s2i = 0
            for i in range(1, len(flags_before)):
                if flags_before[i - 1] == "improving" and flags_before[i] == "stagnating":
                    i2s += 1
                elif flags_before[i - 1] == "stagnating" and flags_before[i] == "improving":
                    s2i += 1

            stage_flag_counts = {st: {"improving": 0, "stagnating": 0} for st in STAGES}
            for st, fl in zip(stage_seq, flags_before):
                stage_flag_counts[st][fl] += 1

            sensitivity_rows.append(
                {
                    "function": fn,
                    "optimizer_seed": seed,
                    "relative_threshold": rel_thr,
                    "stagnating_share": float(np.mean(np.asarray(flags_before) == "stagnating")),
                    "improving_to_stagnating": i2s,
                    "stagnating_to_improving": s2i,
                    "early_improving_count": stage_flag_counts["early"]["improving"],
                    "early_stagnating_count": stage_flag_counts["early"]["stagnating"],
                    "middle_improving_count": stage_flag_counts["middle"]["improving"],
                    "middle_stagnating_count": stage_flag_counts["middle"]["stagnating"],
                    "late_improving_count": stage_flag_counts["late"]["improving"],
                    "late_stagnating_count": stage_flag_counts["late"]["stagnating"],
                }
            )

    # ------------------------------------------------------------------
    # Validation of saved stagnation fields
    # ------------------------------------------------------------------
    counter_mismatch = 0
    flag_mismatch = 0
    for row in transition_rows:
        if row["saved_stagnation_counter_before"] != row["recomputed_stagnation_counter_before"]:
            counter_mismatch += 1
        state_flag = "stagnating" if str(row["state"]).endswith("_stagnating") else "improving"
        if state_flag != row["recomputed_flag_before"]:
            flag_mismatch += 1

    # ------------------------------------------------------------------
    # Aggregations for tables
    # ------------------------------------------------------------------
    trans_by_fn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transition_rows:
        trans_by_fn[row["function"]].append(row)

    run_by_fn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_level_rows:
        run_by_fn[row["function"]].append(row)

    improvement_stats_rows: list[dict[str, Any]] = []
    micro_threshold_rows: list[dict[str, Any]] = []
    for fn in FUNCTIONS:
        rows = trans_by_fn.get(fn, [])
        run_rows = run_by_fn.get(fn, [])
        strict_mask = np.asarray([bool(r["strict_improvement"]) for r in rows], dtype=bool)
        abs_impr = np.asarray([float(r["absolute_improvement"]) for r in rows], dtype=np.float64)
        rel_impr = np.asarray([float(r["relative_improvement"]) for r in rows], dtype=np.float64)
        rel_pos = rel_impr[strict_mask]
        abs_pos = abs_impr[strict_mask]
        if len(abs_pos) == 0:
            abs_pos = np.asarray([0.0], dtype=np.float64)
            rel_pos = np.asarray([0.0], dtype=np.float64)

        improvement_stats_rows.append(
            {
                "function": fn,
                "runs": len(run_rows),
                "transitions": len(rows),
                "strict_improvement_generations_total": int(np.count_nonzero(strict_mask)),
                "no_improvement_generations_total": int(len(rows) - np.count_nonzero(strict_mask)),
                "strict_improvement_ratio": float(np.mean(strict_mask)),
                "mean_positive_abs_improvement": float(np.mean(abs_pos)),
                "median_positive_abs_improvement": float(np.median(abs_pos)),
                "min_positive_abs_improvement": float(np.min(abs_pos)),
                "max_positive_abs_improvement": float(np.max(abs_pos)),
                "mean_positive_rel_improvement": float(np.mean(rel_pos)),
                "median_positive_rel_improvement": float(np.median(rel_pos)),
                "min_positive_rel_improvement": float(np.min(rel_pos)),
                "max_positive_rel_improvement": float(np.max(rel_pos)),
                "mean_stagnating_share_run": float(
                    np.mean([float(r["stagnating_share"]) for r in run_rows]) if run_rows else 0.0
                ),
                "mean_max_non_improvement_streak_run": float(
                    np.mean([float(r["max_non_improvement_streak"]) for r in run_rows]) if run_rows else 0.0
                ),
            }
        )

        for thr in relative_thresholds:
            if len(rel_pos) == 0:
                frac = 0.0
            else:
                frac = float(np.mean(rel_pos < thr))
            micro_threshold_rows.append(
                {
                    "function": fn,
                    "relative_threshold": thr,
                    "fraction_of_strict_improvements_below_threshold": frac,
                }
            )

    sensitivity_summary_rows: list[dict[str, Any]] = []
    for fn in FUNCTIONS:
        fn_rows = [r for r in sensitivity_rows if r["function"] == fn]
        for thr in relative_thresholds:
            thr_rows = [r for r in fn_rows if float(r["relative_threshold"]) == float(thr)]
            if not thr_rows:
                continue
            sensitivity_summary_rows.append(
                {
                    "function": fn,
                    "relative_threshold": thr,
                    "mean_stagnating_share": float(np.mean([r["stagnating_share"] for r in thr_rows])),
                    "mean_improving_to_stagnating": float(
                        np.mean([r["improving_to_stagnating"] for r in thr_rows])
                    ),
                    "mean_stagnating_to_improving": float(
                        np.mean([r["stagnating_to_improving"] for r in thr_rows])
                    ),
                    "mean_early_stagnating_count": float(
                        np.mean([r["early_stagnating_count"] for r in thr_rows])
                    ),
                    "mean_middle_stagnating_count": float(
                        np.mean([r["middle_stagnating_count"] for r in thr_rows])
                    ),
                    "mean_late_stagnating_count": float(
                        np.mean([r["late_stagnating_count"] for r in thr_rows])
                    ),
                }
            )

    state_visit_rows: list[dict[str, Any]] = []
    n_runs = len(run_level_rows)
    for state in STATE6:
        total = visits_by_state_total[state]
        per_run = np.asarray(state_visits_per_run[state], dtype=np.float64)
        non_visited_frac = float(np.mean(per_run == 0.0)) if len(per_run) else 0.0
        action_counts = actions_by_state[state]
        total_actions = sum(action_counts.values())
        action_freqs = {
            a: (action_counts[a] / total_actions if total_actions > 0 else 0.0) for a in ACTIONS
        }
        rewards = rewards_by_state[state]
        state_visit_rows.append(
            {
                "state": state,
                "total_visits": total,
                "mean_visits_per_run": (total / n_runs) if n_runs else 0.0,
                "fraction_runs_not_visited": non_visited_frac,
                "action_freq_rand_1": action_freqs["rand_1"],
                "action_freq_best_1": action_freqs["best_1"],
                "action_freq_current_to_best_1": action_freqs["current_to_best_1"],
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            }
        )

    q_update_rows: list[dict[str, Any]] = []
    for state in STATE6:
        for action in ACTIONS:
            vals = np.asarray(q_updates_per_state_action_per_run[(state, action)], dtype=np.float64)
            q_update_rows.append(
                {
                    "state": state,
                    "action": action,
                    "mean_updates_per_run": float(np.mean(vals)) if len(vals) else 0.0,
                    "median_updates_per_run": float(np.median(vals)) if len(vals) else 0.0,
                    "min_updates_per_run": float(np.min(vals)) if len(vals) else 0.0,
                    "max_updates_per_run": float(np.max(vals)) if len(vals) else 0.0,
                }
            )

    # ------------------------------------------------------------------
    # Save CSV outputs
    # ------------------------------------------------------------------
    _write_csv(
        out_root / "transition_level_metrics.csv",
        transition_rows,
        list(transition_rows[0].keys()),
    )
    _write_csv(
        out_root / "run_level_metrics.csv",
        run_level_rows,
        list(run_level_rows[0].keys()),
    )
    _write_csv(
        out_root / "improvement_stats_by_function.csv",
        improvement_stats_rows,
        list(improvement_stats_rows[0].keys()),
    )
    _write_csv(
        out_root / "micro_improvement_thresholds.csv",
        micro_threshold_rows,
        list(micro_threshold_rows[0].keys()),
    )
    _write_csv(
        out_root / "stagnation_sensitivity_summary.csv",
        sensitivity_summary_rows,
        list(sensitivity_summary_rows[0].keys()),
    )
    _write_csv(
        out_root / "state_visit_statistics.csv",
        state_visit_rows,
        list(state_visit_rows[0].keys()),
    )
    _write_csv(
        out_root / "q_update_distribution.csv",
        q_update_rows,
        list(q_update_rows[0].keys()),
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    # 1) Global best vs FEs (mean and median across 30 runs; log Y).
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    for i, fn in enumerate(FUNCTIONS):
        ax = axes_flat[i]
        hists = []
        for rec in records:
            if str(rec["metadata"]["function_name"]) == fn:
                hists.append(np.asarray(rec["run"]["result"]["best_value_history"], dtype=np.float64))
        arr = np.asarray(hists, dtype=np.float64)
        x = np.arange(1, arr.shape[1] + 1) * pop_size
        mean_curve = np.mean(arr, axis=0)
        median_curve = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        ax.plot(x, mean_curve, label="mean")
        ax.plot(x, median_curve, label="median", linestyle="--")
        ax.fill_between(x, q25, q75, alpha=0.15, label="IQR")
        ax.set_title(fn)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Global best value")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_root / "figure_best_vs_fes_log.png", dpi=160)
    plt.close(fig)

    # 2) Improvement magnitude by transition index (mean absolute/relative; log Y).
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()
    for i, fn in enumerate(FUNCTIONS):
        ax = axes_flat[i]
        rows = trans_by_fn[fn]
        by_t_abs: dict[int, list[float]] = defaultdict(list)
        by_t_rel: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            t = int(r["transition"])
            if float(r["absolute_improvement"]) > 0.0:
                by_t_abs[t].append(float(r["absolute_improvement"]))
                by_t_rel[t].append(max(float(r["relative_improvement"]), 1e-20))
        ts = sorted(by_t_abs.keys())
        x = np.asarray(ts, dtype=np.int64) * pop_size
        y_abs = np.asarray([np.mean(by_t_abs[t]) for t in ts], dtype=np.float64)
        y_rel = np.asarray([np.mean(by_t_rel[t]) for t in ts], dtype=np.float64)
        ax.plot(x, y_abs, label="mean abs improvement")
        ax.plot(x, y_rel, label="mean rel improvement", linestyle="--")
        ax.set_title(fn)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Improvement magnitude")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_root / "figure_improvement_magnitude_vs_fes_log.png", dpi=160)
    plt.close(fig)

    # 3) Distributions of positive improvements (absolute/relative, log10).
    abs_data = []
    rel_data = []
    labels = []
    for fn in FUNCTIONS:
        rows = trans_by_fn[fn]
        abs_vals = [float(r["absolute_improvement"]) for r in rows if float(r["absolute_improvement"]) > 0.0]
        rel_vals = [float(r["relative_improvement"]) for r in rows if float(r["relative_improvement"]) > 0.0]
        if not abs_vals:
            abs_vals = [1e-20]
        if not rel_vals:
            rel_vals = [1e-20]
        abs_data.append(np.log10(np.asarray(abs_vals, dtype=np.float64)))
        rel_data.append(np.log10(np.asarray(rel_vals, dtype=np.float64)))
        labels.append(fn)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].boxplot(abs_data, tick_labels=labels, showfliers=False)
    axes[0].set_title("log10(absolute improvements), positive only")
    axes[0].set_ylabel("log10(value)")
    axes[0].grid(True, alpha=0.25)
    axes[1].boxplot(rel_data, tick_labels=labels, showfliers=False)
    axes[1].set_title("log10(relative improvements), positive only")
    axes[1].set_ylabel("log10(value)")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_root / "figure_improvement_distributions_log10.png", dpi=160)
    plt.close(fig)

    # 4) Stagnation share sensitivity across thresholds.
    threshold_labels = [f"{thr:g}" for thr in relative_thresholds]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(FUNCTIONS), dtype=np.float64)
    width = 0.23
    for j, thr in enumerate(relative_thresholds):
        vals = []
        for fn in FUNCTIONS:
            rows = [
                r
                for r in sensitivity_summary_rows
                if r["function"] == fn and float(r["relative_threshold"]) == float(thr)
            ]
            vals.append(rows[0]["mean_stagnating_share"] * 100.0)
        ax.bar(x + (j - 1) * width, vals, width=width, label=f"rel>{thr:g}")
    # actual strict criterion
    actual = []
    for fn in FUNCTIONS:
        rows = run_by_fn[fn]
        actual.append(float(np.mean([r["stagnating_share"] for r in rows])) * 100.0)
    ax.plot(x, actual, color="black", marker="o", linestyle="--", label="current strict rule")
    ax.set_xticks(x)
    ax.set_xticklabels(FUNCTIONS)
    ax.set_ylabel("Stagnating transitions (%)")
    ax.set_title("Stagnation rate sensitivity (counterfactual, fixed trajectories)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "figure_stagnation_sensitivity.png", dpi=160)
    plt.close(fig)

    # 5) State visitation heatmap (mean visits per run) by function.
    mat = np.zeros((len(FUNCTIONS), len(STATE6)), dtype=np.float64)
    for i, fn in enumerate(FUNCTIONS):
        for j, st in enumerate(STATE6):
            vals = np.asarray(visits_heatmap_by_function_state[fn][st], dtype=np.float64)
            mat[i, j] = float(np.mean(vals)) if len(vals) else 0.0
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(mat, cmap="magma")
    ax.set_xticks(np.arange(len(STATE6)))
    ax.set_xticklabels(STATE6, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FUNCTIONS)))
    ax.set_yticklabels(FUNCTIONS)
    ax.set_title("Mean state visits per run (RL-DE v2)")
    for i in range(len(FUNCTIONS)):
        for j in range(len(STATE6)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean visits per run")
    fig.tight_layout()
    fig.savefig(out_root / "figure_state_visit_heatmap.png", dpi=160)
    plt.close(fig)

    report = {
        "input": {
            "v2_batch_dir": str(v2_batch_dir.resolve()),
            "runs_root": str(runs_root.resolve()),
            "stagnation_threshold": stagnation_threshold,
            "relative_thresholds": list(relative_thresholds),
            "function_scales": function_scales,
            "scale_definition": (
                "function-specific scale = median(abs(initial global best across 30 runs))"
            ),
        },
        "data_checks": {
            "records_found": len(records),
            "runs_per_function": runs_per_function,
            "best_history_lengths_per_function": {
                fn: sorted(set(lengths)) for fn, lengths in best_history_lengths.items()
            },
            "history_recording_inference": (
                "best_value_history length equals configured generations; "
                "with 200 generations and NP=50 this indicates one best value per generation "
                "(including initialization), not per individual evaluation."
            ),
            "budget_fes_detected": sorted(total_fes_detected),
            "generations_detected": sorted(generations_detected),
            "pop_sizes_detected": sorted(pop_sizes_detected),
            "best_monotonic_non_worsening_violations": monotonic_violations,
            "duplicate_function_seed_keys": duplicate_keys,
        },
        "stagnation_recompute_consistency": {
            "counter_mismatches": counter_mismatch,
            "state_flag_mismatches": flag_mismatch,
        },
        "improvement_stats_by_function": improvement_stats_rows,
        "micro_improvement_thresholds": micro_threshold_rows,
        "state_visit_statistics": state_visit_rows,
        "q_update_distribution": q_update_rows,
        "artifacts": {
            "transition_level_metrics_csv": str((out_root / "transition_level_metrics.csv").resolve()),
            "run_level_metrics_csv": str((out_root / "run_level_metrics.csv").resolve()),
            "improvement_stats_csv": str((out_root / "improvement_stats_by_function.csv").resolve()),
            "micro_threshold_csv": str((out_root / "micro_improvement_thresholds.csv").resolve()),
            "sensitivity_csv": str((out_root / "stagnation_sensitivity_summary.csv").resolve()),
            "state_visits_csv": str((out_root / "state_visit_statistics.csv").resolve()),
            "q_update_csv": str((out_root / "q_update_distribution.csv").resolve()),
            "fig_best_vs_fes": str((out_root / "figure_best_vs_fes_log.png").resolve()),
            "fig_improvement_vs_fes": str((out_root / "figure_improvement_magnitude_vs_fes_log.png").resolve()),
            "fig_improvement_distributions": str(
                (out_root / "figure_improvement_distributions_log10.png").resolve()
            ),
            "fig_stagnation_sensitivity": str(
                (out_root / "figure_stagnation_sensitivity.png").resolve()
            ),
            "fig_state_heatmap": str((out_root / "figure_state_visit_heatmap.png").resolve()),
        },
        "limitations": [
            "Counterfactual stagnation analysis uses fixed trajectories from existing runs.",
            "Changing stagnation criterion during optimization would change policy updates and trajectory.",
            "Therefore, counterfactual results are diagnostic only, not causal performance estimates.",
        ],
    }

    (out_root / "diagnostics_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
