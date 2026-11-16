from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rl_optimization.experiments.comparison import load_experiment_config
from rl_optimization.experiments.comparison import run_comparison_experiment
from rl_optimization.experiments.stats import compute_basic_stats
from rl_optimization.experiments.stats import compute_strategy_stage_frequencies_complete
from rl_optimization.experiments.stats import rank_algorithms
from rl_optimization.experiments.stats import rank_algorithms_by_function_means


CONFIG_PATH = Path("experiments/configs/rl_de_comparison_check.yaml")


def test_load_experiment_config_validates_structure() -> None:
    config = load_experiment_config(CONFIG_PATH)
    assert "task" in config and "de" in config and "rl" in config and "experiment" in config


def test_load_experiment_config_rejects_invalid_source(tmp_path: Path) -> None:
    bad = {
        "task": {
            "function_name": {"value": "sphere", "source": "wrong"},
            "dimension": {"value": 2, "source": "assumption"},
            "bounds_by_function": {"value": {"sphere": [-5.12, 5.12]}, "source": "assumption"},
            "minimization": {"value": True, "source": "assumption"},
        },
        "de": {k: {"value": 1, "source": "assumption"} for k in [
            "population_size", "F", "CR", "max_fes", "generations", "crossover", "boundary_handling", "elitism"
        ]},
        "rl": {k: {"value": 1, "source": "assumption"} for k in [
            "method", "alpha", "gamma", "epsilon", "q_init", "state_definition", "actions",
            "action_frequency", "reward_definition", "terminal_update_rule", "reset_q_each_run"
        ]},
        "experiment": {k: {"value": 1, "source": "assumption"} for k in [
            "independent_runs", "base_seed", "seed_rule", "metrics", "std_ddof", "ranking_rule"
        ]},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_load_experiment_config_rejects_invalid_numeric_constraints(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["de"]["population_size"]["value"] = 0
    path = tmp_path / "bad_numeric.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_load_experiment_config_rejects_invalid_budget_divisibility(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["de"]["max_fes"]["value"] = 95
    path = tmp_path / "bad_budget.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_load_experiment_config_rejects_invalid_actions_set(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["rl"]["actions"]["value"] = ["rand_1", "best_1", "rand_1"]
    path = tmp_path / "bad_actions.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_load_experiment_config_rejects_unsupported_seed_rule(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["experiment"]["seed_rule"]["value"] = "unsupported"
    path = tmp_path / "bad_seed_rule.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        run_comparison_experiment(
            config_path=path,
            algorithm="de_rand_1",
            function_name="sphere",
            repeats=1,
            output_dir=tmp_path,
            debug=False,
        )


def test_compute_basic_stats_known_values() -> None:
    vals = [1.0, 2.0, 3.0, 4.0]
    stats = compute_basic_stats(vals, ddof=1)
    assert np.isclose(stats["mean"], 2.5)
    assert np.isclose(stats["std"], np.std(np.array(vals), ddof=1))
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0


def test_rank_algorithms_with_exact_ties() -> None:
    results = {
        "a": [1.0, 2.0],
        "b": [1.0, 1.0],
        "c": [3.0, 2.0],
    }
    ranked = rank_algorithms(results, minimization=True)
    assert np.isclose(ranked["ranks_by_algorithm"]["a"][0], 1.5)
    assert np.isclose(ranked["ranks_by_algorithm"]["b"][0], 1.5)
    assert np.isclose(ranked["ranks_by_algorithm"]["c"][0], 3.0)


def test_rank_algorithms_by_function_means_known_case() -> None:
    function_results = {
        "sphere": {
            "alg_a": [1.0, 1.0, 1.0],
            "alg_b": [2.0, 2.0, 2.0],
            "alg_c": [3.0, 3.0, 3.0],
        },
        "ackley": {
            "alg_a": [3.0, 3.0, 3.0],
            "alg_b": [1.0, 1.0, 1.0],
            "alg_c": [2.0, 2.0, 2.0],
        },
    }
    ranked = rank_algorithms_by_function_means(function_results, minimization=True, ddof=1)
    assert np.isclose(ranked["per_function_ranks"]["sphere"]["alg_a"], 1.0)
    assert np.isclose(ranked["per_function_ranks"]["ackley"]["alg_b"], 1.0)
    assert np.isclose(ranked["mean_rank"]["alg_a"], 2.0)
    assert np.isclose(ranked["mean_rank"]["alg_b"], 1.5)
    assert ranked["tie_rule"] == "exact_equality_on_mean_values"


def test_strategy_stage_frequencies_complete_has_all_states_actions() -> None:
    histories = [
        [{"state": "early", "action": "rand_1"}],
        [{"state": "middle", "action": "best_1"}],
    ]
    freq = compute_strategy_stage_frequencies_complete(histories, as_percent=False)
    assert set(freq.keys()) == {"early", "middle", "late"}
    for state in ("early", "middle", "late"):
        assert set(freq[state].keys()) == {"rand_1", "best_1", "current_to_best_1"}
    assert np.isclose(freq["late"]["rand_1"], 0.0)
    freq_percent = compute_strategy_stage_frequencies_complete(histories, as_percent=True)
    assert np.isclose(freq_percent["early"]["rand_1"], 100.0)


def test_run_comparison_uses_function_specific_bounds(tmp_path: Path) -> None:
    out = run_comparison_experiment(
        config_path=CONFIG_PATH,
        algorithm="de_rand_1",
        function_name="schwefel",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    data = json.loads(Path(out["output_path"]).read_text(encoding="utf-8"))
    assert data["metadata"]["actual_bounds"] == [-500.0, 500.0]
    cfg = data["runs"][0]["result"]["config"]
    assert cfg["left_border"] == -500.0
    assert cfg["right_border"] == 500.0


def test_run_comparison_saves_params_and_seeds(tmp_path: Path) -> None:
    out = run_comparison_experiment(
        config_path=CONFIG_PATH,
        algorithm="rl_de_q_learning",
        function_name="sphere",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    output_path = Path(out["output_path"])
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["metadata"]["algorithm"] == "rl_de_q_learning"
    run0 = data["runs"][0]
    assert run0["optimizer_seed"] == 42
    assert run0["rl_seed"] == 42
    assert len(run0["result"]["best_value_history"]) == 10
    assert len(run0["result"]["strategy_history"]) == 10
    assert run0["result"]["function_evaluations"] == 100


def test_run_comparison_rejects_rl_carry_learning_in_comparison_mode(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["rl"]["reset_q_each_run"]["value"] = False
    path = tmp_path / "bad_reset.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        run_comparison_experiment(
            config_path=path,
            algorithm="rl_de_q_learning",
            function_name="sphere",
            repeats=1,
            output_dir=tmp_path,
            debug=False,
        )


def test_run_comparison_budget_matches_max_fes(tmp_path: Path) -> None:
    out = run_comparison_experiment(
        config_path=CONFIG_PATH,
        algorithm="de_rand_1",
        function_name="sphere",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    output_path = Path(out["output_path"])
    data = json.loads(output_path.read_text(encoding="utf-8"))
    max_fes = data["config"]["de"]["max_fes"]["value"]
    used = data["runs"][0]["result"]["function_evaluations"]
    assert used == max_fes


def test_run_comparison_saves_random_de_selector_seed(tmp_path: Path) -> None:
    out = run_comparison_experiment(
        config_path=CONFIG_PATH,
        algorithm="random_de",
        function_name="sphere",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    data = json.loads(Path(out["output_path"]).read_text(encoding="utf-8"))
    run0 = data["runs"][0]
    assert run0["strategy_selector_seed"] == 10042


def test_run_comparison_does_not_overwrite_previous_results(tmp_path: Path) -> None:
    out1 = run_comparison_experiment(
        config_path=CONFIG_PATH,
        algorithm="de_best_1",
        function_name="sphere",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    out2 = run_comparison_experiment(
        config_path=CONFIG_PATH,
        algorithm="de_best_1",
        function_name="sphere",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    assert out1["output_path"] != out2["output_path"]
    assert Path(out1["output_path"]).exists()
    assert Path(out2["output_path"]).exists()


def test_load_experiment_config_rejects_missing_stagnation_threshold_for_stagnation_state(
    tmp_path: Path,
) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["rl"]["state_definition"]["value"] = "budget_ratio_thirds_with_stagnation"
    cfg["rl"].pop("stagnation_threshold", None)
    path = tmp_path / "missing_stagnation_threshold.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(path)


def test_run_comparison_supports_rl_de_stagnation_algorithm(tmp_path: Path) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["task"]["dimension"]["value"] = 10
    cfg["de"]["population_size"]["value"] = 50
    cfg["de"]["max_fes"]["value"] = 10000
    cfg["de"]["generations"]["value"] = 200
    cfg["rl"]["state_definition"]["value"] = "budget_ratio_thirds_with_stagnation"
    cfg["rl"]["stagnation_threshold"] = {"value": 5, "source": "assumption"}
    cfg_path = tmp_path / "stagnation_cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    out = run_comparison_experiment(
        config_path=cfg_path,
        algorithm="rl_de_q_learning_stagnation",
        function_name="sphere",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
    )
    data = json.loads(Path(out["output_path"]).read_text(encoding="utf-8"))
    run0 = data["runs"][0]
    assert data["metadata"]["algorithm"] == "rl_de_q_learning_stagnation"
    assert run0["result"]["function_evaluations"] == 10000
    assert len(run0["result"]["rl_history"]) == 199


def test_run_comparison_rejects_stagnation_algorithm_with_v1_state_definition(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_comparison_experiment(
            config_path=CONFIG_PATH,
            algorithm="rl_de_q_learning_stagnation",
            function_name="sphere",
            repeats=1,
            output_dir=tmp_path,
            debug=False,
        )
