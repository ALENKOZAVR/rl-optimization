from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rl_optimization.algorithms.differential_evolution import DEConfig
from rl_optimization.algorithms.differential_evolution import DEDebugConfig
from rl_optimization.algorithms.differential_evolution import RL_STATE_DEFINITION_BUDGET_THIRDS
from rl_optimization.algorithms.differential_evolution import RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION
from rl_optimization.algorithms.differential_evolution import RLConfig
from rl_optimization.algorithms.differential_evolution import run_differential_evolution
from rl_optimization.algorithms.differential_evolution import run_rl_differential_evolution
from rl_optimization.benchmarks import BENCHMARK_FUNCTIONS


VALID_SOURCES = {"article", "existing_implementation", "assumption"}
SUPPORTED_CROSSOVERS = {"binomial"}
SUPPORTED_BOUNDARY_HANDLING = {"clip_to_bounds"}
SUPPORTED_ACTIONS = {"rand_1", "best_1", "current_to_best_1"}
SUPPORTED_STATE_DEFINITIONS = {
    RL_STATE_DEFINITION_BUDGET_THIRDS,
    RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
}
SUPPORTED_REWARD_DEFINITIONS = {"strict_improvement_ratio_per_generation"}
SUPPORTED_ACTION_FREQUENCY = {"one_action_per_generation"}
SUPPORTED_TERMINAL_UPDATE = {"done_true_on_last_generation_transition"}
SUPPORTED_RL_METHODS = {"q_learning"}
SUPPORTED_Q_INIT = {"zeros"}
ALGORITHM_ALIASES = {
    "de_rand_1": "rand_1",
    "de_best_1": "best_1",
    "de_current_to_best_1": "current_to_best_1",
    "random_de": "random_de",
    "rl_de_q_learning": "rl_de_q_learning",
    "rl_de_q_learning_stagnation": "rl_de_q_learning_stagnation",
}


@dataclass(frozen=True)
class ParamWithSource:
    value: Any
    source: str


def _parse_param(config: dict[str, Any], key: str) -> ParamWithSource:
    if key not in config:
        raise ValueError(f"Missing required config key: {key}")
    raw = config[key]
    if not isinstance(raw, dict) or "value" not in raw or "source" not in raw:
        raise ValueError(f"Config key '{key}' must contain 'value' and 'source'")
    source = str(raw["source"])
    if source not in VALID_SOURCES:
        raise ValueError(f"Config key '{key}' has invalid source '{source}'")
    return ParamWithSource(value=raw["value"], source=source)


def _parse_optional_param(config: dict[str, Any], key: str) -> ParamWithSource | None:
    if key not in config:
        return None
    return _parse_param(config, key)


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    for section in ("task", "de", "rl", "experiment"):
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"Config must include section '{section}'")

    task = data["task"]
    function_name_default = _parse_param(task, "function_name").value
    if function_name_default not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown benchmark function '{function_name_default}'")

    dimension = int(_parse_param(task, "dimension").value)
    if dimension <= 0:
        raise ValueError("task.dimension must be > 0")

    minimization = _parse_param(task, "minimization").value
    if not isinstance(minimization, bool):
        raise ValueError("task.minimization must be bool")

    bounds_by_function = _parse_param(task, "bounds_by_function").value
    if not isinstance(bounds_by_function, dict):
        raise ValueError("task.bounds_by_function must be a mapping function -> [left, right]")
    for fn_name in BENCHMARK_FUNCTIONS:
        if fn_name not in bounds_by_function:
            raise ValueError(f"task.bounds_by_function is missing '{fn_name}'")
        bounds = bounds_by_function[fn_name]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"Bounds for '{fn_name}' must be [left, right]")
        left, right = float(bounds[0]), float(bounds[1])
        if not np.isfinite(left) or not np.isfinite(right):
            raise ValueError(f"Bounds for '{fn_name}' must be finite")
        if left >= right:
            raise ValueError(f"Bounds for '{fn_name}' must satisfy left < right")

    de = data["de"]
    for key in (
        "population_size",
        "F",
        "CR",
        "max_fes",
        "generations",
        "crossover",
        "boundary_handling",
        "elitism",
    ):
        _parse_param(de, key)
    pop_size = int(_parse_param(de, "population_size").value)
    if pop_size < 4:
        raise ValueError("de.population_size must be >= 4")
    f = float(_parse_param(de, "F").value)
    if f <= 0:
        raise ValueError("de.F must be > 0")
    cr = float(_parse_param(de, "CR").value)
    if not (0.0 <= cr <= 1.0):
        raise ValueError("de.CR must be in [0, 1]")
    max_fes = int(_parse_param(de, "max_fes").value)
    if max_fes <= 0:
        raise ValueError("de.max_fes must be > 0")
    if max_fes % pop_size != 0:
        raise ValueError("de.max_fes must be divisible by de.population_size")
    generations = int(_parse_param(de, "generations").value)
    if generations <= 0:
        raise ValueError("de.generations must be > 0")
    expected_generations = max_fes // pop_size
    if generations != expected_generations:
        raise ValueError(
            f"de.generations ({generations}) must equal de.max_fes/de.population_size ({expected_generations})"
        )
    crossover = str(_parse_param(de, "crossover").value)
    if crossover not in SUPPORTED_CROSSOVERS:
        raise ValueError(f"Unsupported de.crossover '{crossover}'")
    boundary = str(_parse_param(de, "boundary_handling").value)
    if boundary not in SUPPORTED_BOUNDARY_HANDLING:
        raise ValueError(f"Unsupported de.boundary_handling '{boundary}'")
    if not isinstance(_parse_param(de, "elitism").value, bool):
        raise ValueError("de.elitism must be bool")

    rl = data["rl"]
    for key in (
        "method",
        "alpha",
        "gamma",
        "epsilon",
        "q_init",
        "state_definition",
        "actions",
        "action_frequency",
        "reward_definition",
        "terminal_update_rule",
        "reset_q_each_run",
    ):
        _parse_param(rl, key)
    if str(_parse_param(rl, "method").value) not in SUPPORTED_RL_METHODS:
        raise ValueError("rl.method must be q_learning")
    alpha = float(_parse_param(rl, "alpha").value)
    gamma = float(_parse_param(rl, "gamma").value)
    epsilon = float(_parse_param(rl, "epsilon").value)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("rl.alpha must be in [0, 1]")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("rl.gamma must be in [0, 1]")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("rl.epsilon must be in [0, 1]")
    if str(_parse_param(rl, "q_init").value) not in SUPPORTED_Q_INIT:
        raise ValueError("rl.q_init must be zeros")
    if str(_parse_param(rl, "state_definition").value) not in SUPPORTED_STATE_DEFINITIONS:
        raise ValueError("Unsupported rl.state_definition")
    actions = _parse_param(rl, "actions").value
    if not isinstance(actions, list) or len(actions) != 3:
        raise ValueError("rl.actions must be a list of three actions")
    if set(actions) != SUPPORTED_ACTIONS:
        raise ValueError(f"rl.actions must contain exactly {sorted(SUPPORTED_ACTIONS)}")
    if str(_parse_param(rl, "action_frequency").value) not in SUPPORTED_ACTION_FREQUENCY:
        raise ValueError("Unsupported rl.action_frequency")
    if str(_parse_param(rl, "reward_definition").value) not in SUPPORTED_REWARD_DEFINITIONS:
        raise ValueError("Unsupported rl.reward_definition")
    if str(_parse_param(rl, "terminal_update_rule").value) not in SUPPORTED_TERMINAL_UPDATE:
        raise ValueError("Unsupported rl.terminal_update_rule")
    if not isinstance(_parse_param(rl, "reset_q_each_run").value, bool):
        raise ValueError("rl.reset_q_each_run must be bool")

    state_definition = str(_parse_param(rl, "state_definition").value)
    stag_param = _parse_optional_param(rl, "stagnation_threshold")
    if stag_param is not None:
        stagnation_threshold = int(stag_param.value)
        if stagnation_threshold <= 0:
            raise ValueError("rl.stagnation_threshold must be > 0")
    if state_definition == RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION and stag_param is None:
        raise ValueError(
            "rl.stagnation_threshold must be provided for "
            "state_definition='budget_ratio_thirds_with_stagnation'"
        )

    experiment = data["experiment"]
    for key in (
        "independent_runs",
        "base_seed",
        "seed_rule",
        "metrics",
        "std_ddof",
        "ranking_rule",
    ):
        _parse_param(experiment, key)
    independent_runs = int(_parse_param(experiment, "independent_runs").value)
    if independent_runs < 1:
        raise ValueError("experiment.independent_runs must be >= 1")
    base_seed = int(_parse_param(experiment, "base_seed").value)
    if base_seed < 0:
        raise ValueError("experiment.base_seed must be >= 0")
    ddof = int(_parse_param(experiment, "std_ddof").value)
    if ddof not in (0, 1):
        raise ValueError("experiment.std_ddof must be 0 or 1")
    metrics = _parse_param(experiment, "metrics").value
    if not isinstance(metrics, list) or len(metrics) == 0:
        raise ValueError("experiment.metrics must be a non-empty list")

    return data


def _get_version_safe(package_name: str) -> str:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "unknown"


def run_comparison_experiment(
    config_path: str | Path,
    algorithm: str,
    function_name: str,
    repeats: int | None = None,
    output_dir: str | Path = "experiment_results",
    debug: bool = False,
    run_offset: int = 0,
    enforce_reset_q_each_run_for_comparison: bool = True,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if run_offset < 0:
        raise ValueError("run_offset must be >= 0")
    config_independent_runs = int(_parse_param(config["experiment"], "independent_runs").value)
    repeats_to_use = config_independent_runs if repeats is None else repeats
    if repeats_to_use < 1:
        raise ValueError("repeats must be >= 1")
    algo = ALGORITHM_ALIASES.get(algorithm, algorithm)
    if algo not in {
        "rand_1",
        "best_1",
        "current_to_best_1",
        "random_de",
        "rl_de_q_learning",
        "rl_de_q_learning_stagnation",
    }:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")
    if function_name not in BENCHMARK_FUNCTIONS:
        raise ValueError(f"Unknown function '{function_name}'")

    task = config["task"]
    de_cfg = config["de"]
    rl_cfg = config["rl"]
    exp_cfg = config["experiment"]

    dimension = int(_parse_param(task, "dimension").value)
    bounds_by_function = _parse_param(task, "bounds_by_function").value
    minimization = bool(_parse_param(task, "minimization").value)
    bounds = bounds_by_function[function_name]
    left, right = float(bounds[0]), float(bounds[1])

    pop_size = int(_parse_param(de_cfg, "population_size").value)
    f = float(_parse_param(de_cfg, "F").value)
    cr = float(_parse_param(de_cfg, "CR").value)
    max_fes = int(_parse_param(de_cfg, "max_fes").value)
    elitism = bool(_parse_param(de_cfg, "elitism").value)
    generations = int(_parse_param(de_cfg, "generations").value)
    if max_fes != generations * pop_size:
        raise ValueError("Inconsistent DE budget: max_fes must equal generations * population_size")
    iters = generations

    rl_alpha = float(_parse_param(rl_cfg, "alpha").value)
    rl_gamma = float(_parse_param(rl_cfg, "gamma").value)
    rl_epsilon = float(_parse_param(rl_cfg, "epsilon").value)
    rl_state_definition = str(_parse_param(rl_cfg, "state_definition").value)
    rl_stagnation_threshold_param = _parse_optional_param(rl_cfg, "stagnation_threshold")
    rl_stagnation_threshold = (
        int(rl_stagnation_threshold_param.value)
        if rl_stagnation_threshold_param is not None
        else 5
    )
    reset_q_each_run = bool(_parse_param(rl_cfg, "reset_q_each_run").value)

    base_seed = int(_parse_param(exp_cfg, "base_seed").value)
    seed_rule = str(_parse_param(exp_cfg, "seed_rule").value).strip()
    if seed_rule != "seed_i = base_seed + run_index":
        raise ValueError(
            "Unsupported seed_rule. Current implementation supports only "
            "'seed_i = base_seed + run_index'."
        )

    benchmark_fn = BENCHMARK_FUNCTIONS[function_name]
    debug_cfg = DEDebugConfig(enabled=debug, generation_to_trace=1, individual_index_to_trace=0)

    if (
        algo in {"rl_de_q_learning", "rl_de_q_learning_stagnation"}
        and enforce_reset_q_each_run_for_comparison
        and not reset_q_each_run
    ):
        raise ValueError(
            "For comparison mode, rl.reset_q_each_run must be true. "
            "Use a separate experiment type for transfer learning."
        )
    if algo == "rl_de_q_learning" and rl_state_definition != RL_STATE_DEFINITION_BUDGET_THIRDS:
        raise ValueError(
            "Algorithm 'rl_de_q_learning' requires rl.state_definition='budget_ratio_thirds'."
        )
    if (
        algo == "rl_de_q_learning_stagnation"
        and rl_state_definition != RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION
    ):
        raise ValueError(
            "Algorithm 'rl_de_q_learning_stagnation' requires "
            "rl.state_definition='budget_ratio_thirds_with_stagnation'."
        )

    runs_data: list[dict[str, Any]] = []
    for local_idx in range(repeats_to_use):
        run_index = run_offset + local_idx
        optimizer_seed = base_seed + run_index
        de_config = DEConfig(
            num_variables=dimension,
            left_border=left,
            right_border=right,
            pop_size=pop_size,
            F=f,
            CR=cr,
            iters=iters,
            seed=optimizer_seed,
            mutation="rand_1" if algo in {"rl_de_q_learning", "rl_de_q_learning_stagnation"} else algo,
            objective_mode="population",
            minimization=minimization,
            elitism=elitism,
            strategy_random_seed=optimizer_seed + 10_000,
        )

        if algo in {"rl_de_q_learning", "rl_de_q_learning_stagnation"}:
            rl_config = RLConfig(
                alpha=rl_alpha,
                gamma=rl_gamma,
                epsilon=rl_epsilon,
                seed=base_seed + run_index,
                carry_learning=not reset_q_each_run,
                state_definition=rl_state_definition,
                stagnation_threshold=rl_stagnation_threshold,
            )
            result = run_rl_differential_evolution(
                objective=benchmark_fn,
                config=de_config,
                rl_config=rl_config,
                debug_config=debug_cfg,
            )
            rl_seed = rl_config.seed
            strategy_seed = None
        else:
            result = run_differential_evolution(
                objective=benchmark_fn,
                config=de_config,
                debug_config=debug_cfg,
            )
            rl_seed = None
            strategy_seed = de_config.strategy_random_seed if algo == "random_de" else None

        runs_data.append(
            {
                "run_index": run_index,
                "optimizer_seed": optimizer_seed,
                "rl_seed": rl_seed,
                "strategy_selector_seed": strategy_seed,
                "result": result.to_dict(),
            }
        )

    metadata = {
        "algorithm": algo,
        "function_name": function_name,
        "config_path": str(Path(config_path).resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thefittest_version": _get_version_safe("thefittest"),
        "git_commit": None,
        "used_repeats": repeats_to_use,
        "run_offset": run_offset,
        "actual_bounds": [left, right],
        "actual_dimension": dimension,
        "seed_rule": seed_rule,
    }

    payload = {
        "metadata": metadata,
        "config": config,
        "runs": runs_data,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{algo}__{function_name}__rep{repeats_to_use}__seed{base_seed}__off{run_offset}"
    output_path = out_dir / f"{base_name}.json"
    if output_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = out_dir / f"{base_name}__{ts}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "output_path": str(output_path),
        "payload": payload,
    }
