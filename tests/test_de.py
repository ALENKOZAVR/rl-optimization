from __future__ import annotations

import numpy as np
import pytest
from thefittest.optimizers import DifferentialEvolution
from thefittest.utils.random import check_random_state

from rl_optimization.algorithms.differential_evolution import best_1_mutation_from_indices
from rl_optimization.algorithms.differential_evolution import current_to_best_1_mutation_from_indices
from rl_optimization.algorithms.differential_evolution import DEConfig
from rl_optimization.algorithms.differential_evolution import DEDebugConfig
from rl_optimization.algorithms.differential_evolution import normalize_mutation_strategy
from rl_optimization.algorithms.differential_evolution import TracingDifferentialEvolution
from rl_optimization.algorithms.differential_evolution import clip_to_bounds
from rl_optimization.algorithms.differential_evolution import rand_1_mutation_from_indices
from rl_optimization.algorithms.differential_evolution import run_differential_evolution
from rl_optimization.algorithms.differential_evolution import should_select_trial_minimization
from rl_optimization.algorithms.differential_evolution import uniform_random_strategy_selector
from rl_optimization.benchmarks.sphere import sphere


def test_sphere_values() -> None:
    population = np.array([[0.0, 0.0], [1.0, 2.0], [-3.0, 4.0]], dtype=np.float64)
    values = sphere(population)
    assert np.allclose(values, np.array([0.0, 5.0, 25.0], dtype=np.float64))


def test_rand_1_formula_with_hand_checkable_numbers() -> None:
    population = np.array(
        [
            [1.0, 1.0],   # 0
            [3.0, -1.0],  # 1
            [0.0, 2.0],   # 2
            [2.0, 5.0],   # 3
        ],
        dtype=np.float64,
    )
    # v = x_r3 + F * (x_r1 - x_r2), r1=1, r2=2, r3=0, F=0.5
    # x_r1 - x_r2 = [3,-1] - [0,2] = [3,-3]
    # F * (...) = [1.5,-1.5]
    # v = [1,1] + [1.5,-1.5] = [2.5,-0.5]
    mutant = rand_1_mutation_from_indices(population, r1=1, r2=2, r3=0, F=0.5)
    assert np.allclose(mutant, np.array([2.5, -0.5], dtype=np.float64))


def test_best_1_formula_with_hand_checkable_numbers() -> None:
    population = np.array(
        [
            [1.0, 1.0],   # 0
            [3.0, -1.0],  # 1
            [0.0, 2.0],   # 2
            [2.0, 5.0],   # 3
        ],
        dtype=np.float64,
    )
    best_vector = np.array([0.5, -0.5], dtype=np.float64)
    # v = x_best + F * (x_r1 - x_r2), r1=1, r2=2, F=0.5
    # x_r1 - x_r2 = [3,-1] - [0,2] = [3,-3]
    # F*(...) = [1.5,-1.5], v=[2.0,-2.0]
    mutant = best_1_mutation_from_indices(best_vector, population, r1=1, r2=2, F=0.5)
    assert np.allclose(mutant, np.array([2.0, -2.0], dtype=np.float64))


def test_current_to_best_1_formula_with_hand_checkable_numbers() -> None:
    population = np.array(
        [
            [1.0, 1.0],   # 0
            [3.0, -1.0],  # 1
            [0.0, 2.0],   # 2
            [2.0, 5.0],   # 3
        ],
        dtype=np.float64,
    )
    current = np.array([2.0, 1.0], dtype=np.float64)
    best = np.array([0.0, 0.0], dtype=np.float64)
    # v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2), r1=1, r2=2, F=0.5
    # F*(x_best - x_i) = [-1.0, -0.5]
    # F*(x_r1 - x_r2) = [1.5, -1.5]
    # v = [2,1] + [-1,-0.5] + [1.5,-1.5] = [2.5,-1.0]
    mutant = current_to_best_1_mutation_from_indices(current, best, population, r1=1, r2=2, F=0.5)
    assert np.allclose(mutant, np.array([2.5, -1.0], dtype=np.float64))


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("rand_1", "rand_1"),
        ("rand/1", "rand_1"),
        ("rand/1/bin", "rand_1"),
        ("de/rand/1/bin", "rand_1"),
        ("best_1", "best_1"),
        ("best/1/bin", "best_1"),
        ("current_to_best_1", "current_to_best_1"),
        ("current-to-best/1/bin", "current_to_best_1"),
        ("random-de", "random_de"),
        ("random_de", "random_de"),
    ],
)
def test_mutation_strategy_aliases(alias: str, expected: str) -> None:
    assert normalize_mutation_strategy(alias) == expected


def test_unknown_mutation_strategy_raises_error() -> None:
    with pytest.raises(ValueError):
        normalize_mutation_strategy("rand/2")


def test_selection_rule_for_minimization() -> None:
    assert should_select_trial_minimization(target_value=10.0, trial_value=8.0)
    assert not should_select_trial_minimization(target_value=10.0, trial_value=12.0)


def test_objective_mode_individual_counts_exact_calls_without_extra_probes() -> None:
    call_counter = {"count": 0}

    def counted_objective(individual: np.ndarray) -> float:
        call_counter["count"] += 1
        return float(np.sum(individual**2))

    cfg = DEConfig(seed=42, pop_size=10, iters=10, mutation="rand_1", objective_mode="individual")
    result = run_differential_evolution(counted_objective, cfg, DEDebugConfig(enabled=False))

    assert result.function_evaluations == cfg.pop_size * cfg.iters
    assert call_counter["count"] == result.function_evaluations


def test_objective_mode_population_counts_match_budget_without_extra_probes() -> None:
    eval_counter = {"count": 0}

    def counted_population_objective(population: np.ndarray) -> np.ndarray:
        eval_counter["count"] += int(len(population))
        return np.sum(population**2, axis=1, dtype=np.float64)

    cfg = DEConfig(seed=42, pop_size=10, iters=10, mutation="best_1", objective_mode="population")
    result = run_differential_evolution(counted_population_objective, cfg, DEDebugConfig(enabled=False))

    assert result.function_evaluations == cfg.pop_size * cfg.iters
    assert eval_counter["count"] == result.function_evaluations


def test_invalid_objective_mode_raises_error() -> None:
    cfg = DEConfig(objective_mode="auto")
    with pytest.raises(ValueError):
        run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))


def test_bounds_control() -> None:
    vector = np.array([-10.0, 0.1, 9.0], dtype=np.float64)
    left = np.array([-5.12, -5.12, -5.12], dtype=np.float64)
    right = np.array([5.12, 5.12, 5.12], dtype=np.float64)
    clipped = clip_to_bounds(vector, left, right)
    assert np.all(clipped >= left)
    assert np.all(clipped <= right)
    assert np.allclose(clipped, np.array([-5.12, 0.1, 5.12], dtype=np.float64))


def test_population_size_preserved() -> None:
    cfg = DEConfig(pop_size=10, iters=5, seed=42)
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    assert len(result.population_history) == cfg.iters
    assert all(pop.shape == (cfg.pop_size, cfg.num_variables) for pop in result.population_history)


def test_function_evaluations_count() -> None:
    cfg = DEConfig(pop_size=10, iters=10, seed=42)
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    assert result.function_evaluations == cfg.pop_size * cfg.iters


def test_reproducibility_with_same_seed() -> None:
    cfg = DEConfig(seed=42)
    result_1 = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    result_2 = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    assert np.allclose(result_1.best_solution, result_2.best_solution)
    assert np.isclose(result_1.best_value, result_2.best_value)
    assert np.allclose(result_1.best_value_history, result_2.best_value_history)


def test_global_best_history_does_not_worsen_with_elitism() -> None:
    cfg = DEConfig(elitism=True, seed=42)
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    history = np.array(result.best_value_history, dtype=np.float64)
    deltas = np.diff(history)
    assert np.all(deltas <= 1e-12)


def test_debug_tracing_matches_thefittest_behavior_for_rand1_bin() -> None:
    # Проверка: трассирующий класс должен давать тот же итог, что и DifferentialEvolution.
    init_population = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )

    common_kwargs = dict(
        fitness_function=sphere,
        iters=2,
        pop_size=5,
        left_border=-5.12,
        right_border=5.12,
        num_variables=2,
        mutation="rand_1",
        F=0.5,
        CR=0.9,
        elitism=True,
        minimization=True,
        keep_history=True,
        init_population=init_population,
        random_state=123,
    )

    base = DifferentialEvolution(**common_kwargs)
    traced = TracingDifferentialEvolution(
        **common_kwargs,
        debug_config=DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )
    base.fit()
    traced.fit()

    assert np.allclose(base._population_g_i, traced._population_g_i)
    assert np.allclose(base.get_stats()["max_fitness"], traced.get_stats()["max_fitness"])
    assert traced.debug_trace is not None


def test_debug_trace_indices_and_mutation_formula_are_consistent() -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="rand_1", F=0.5, CR=0.9)
    result = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )

    assert result.debug_trace is not None
    trace = result.debug_trace["mutation_crossover_selection"]
    snapshot = np.asarray(trace["population_snapshot_before_mutation"], dtype=np.float64)
    r1, r2, r3 = trace["rand_indices"]

    x_r1 = snapshot[r1]
    x_r2 = snapshot[r2]
    x_r3 = snapshot[r3]

    assert np.allclose(x_r1, np.asarray(trace["rand_vectors"]["x_r1"], dtype=np.float64))
    assert np.allclose(x_r2, np.asarray(trace["rand_vectors"]["x_r2"], dtype=np.float64))
    assert np.allclose(x_r3, np.asarray(trace["rand_vectors"]["x_r3"], dtype=np.float64))

    expected_mutant = x_r3 + cfg.F * (x_r1 - x_r2)
    assert np.allclose(expected_mutant, np.asarray(trace["mutant_vector"], dtype=np.float64))
    assert trace["consistency_checks"]["rand_index_vectors_match_snapshot"]
    assert trace["consistency_checks"]["mutant_matches_formula"]


def test_debug_mode_does_not_change_de_result() -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="rand_1", F=0.5, CR=0.9)
    normal = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    debug = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )

    assert np.allclose(normal.best_solution, debug.best_solution)
    assert np.isclose(normal.best_value, debug.best_value)
    assert np.allclose(normal.best_value_history, debug.best_value_history)


def test_full_run_uses_all_10_generations_and_history_matches() -> None:
    cfg = DEConfig(
        num_variables=2,
        left_border=-5.12,
        right_border=5.12,
        pop_size=10,
        F=0.5,
        CR=0.9,
        iters=10,
        seed=42,
        mutation="rand_1",
        elitism=True,
        minimization=True,
    )
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))

    assert len(result.population_history) == cfg.iters
    assert len(result.best_value_history) == cfg.iters

    generation_best = np.array([np.min(sphere(pop)) for pop in result.population_history], dtype=np.float64)
    cumulative_best = np.minimum.accumulate(generation_best)
    assert np.allclose(result.best_value_history, cumulative_best)
    assert np.isclose(result.best_value, result.best_value_history[-1])


@pytest.mark.parametrize("mutation", ["rand_1", "best_1", "current_to_best_1"])
def test_each_strategy_runs_10_generations_and_has_monotonic_best_history(mutation: str) -> None:
    cfg = DEConfig(
        num_variables=2,
        left_border=-5.12,
        right_border=5.12,
        pop_size=10,
        F=0.5,
        CR=0.9,
        iters=10,
        seed=42,
        mutation=mutation,
        elitism=True,
        minimization=True,
    )
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    assert len(result.population_history) == 10
    assert len(result.best_value_history) == 10
    assert np.all(np.diff(np.array(result.best_value_history, dtype=np.float64)) <= 1e-12)


def test_function_evaluations_include_initialization_and_all_generations() -> None:
    cfg = DEConfig(pop_size=10, iters=10, seed=42)
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))

    # 1) Инициализация: pop_size вычислений
    # 2) Ещё (iters-1) поколений, каждое по pop_size вычислений
    expected_calls = cfg.pop_size + (cfg.iters - 1) * cfg.pop_size
    assert result.function_evaluations == expected_calls


@pytest.mark.parametrize("mutation", ["rand_1", "best_1", "current_to_best_1"])
def test_all_individuals_stay_within_bounds_after_boundary_handling(mutation: str) -> None:
    cfg = DEConfig(
        pop_size=10,
        iters=10,
        seed=42,
        left_border=-5.12,
        right_border=5.12,
        mutation=mutation,
        F=0.5,
        CR=0.9,
    )
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))

    left = cfg.left_border - 1e-12
    right = cfg.right_border + 1e-12
    for pop in result.population_history:
        assert np.all(pop >= left)
        assert np.all(pop <= right)


def test_repeated_runs_with_same_seed_are_identical_for_full_run() -> None:
    cfg = DEConfig(pop_size=10, iters=10, seed=42, mutation="rand_1", F=0.5, CR=0.9)
    runs = [run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False)) for _ in range(3)]

    ref = runs[0]
    for run in runs[1:]:
        assert np.allclose(run.best_solution, ref.best_solution)
        assert np.isclose(run.best_value, ref.best_value)
        assert np.allclose(run.best_value_history, ref.best_value_history)
        assert run.function_evaluations == ref.function_evaluations


@pytest.mark.parametrize("mutation", ["rand_1", "best_1", "current_to_best_1"])
def test_debug_trace_consistency_for_each_supported_strategy(mutation: str) -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation=mutation, F=0.5, CR=0.9)
    result = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )

    assert result.debug_trace is not None
    trace = result.debug_trace["mutation_crossover_selection"]
    checks = trace["consistency_checks"]
    assert checks["rand_indices_unique"]
    assert checks["rand_indices_in_range"]
    assert checks["rand_vector_keys_match_strategy"]
    assert checks["rand_index_vectors_match_snapshot"]
    assert checks["mutant_matches_formula"]

    if mutation == "rand_1":
        assert len(trace["rand_indices"]) == 3
    else:
        assert len(trace["rand_indices"]) == 2
        assert "best_vector_before_mutation" in trace


@pytest.mark.parametrize("mutation", ["rand_1", "best_1", "current_to_best_1"])
def test_normal_and_debug_match_for_each_supported_strategy(mutation: str) -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation=mutation, F=0.5, CR=0.9)
    normal = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    debug = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )

    assert np.allclose(normal.best_solution, debug.best_solution)
    assert np.isclose(normal.best_value, debug.best_value)
    assert np.allclose(normal.best_value_history, debug.best_value_history)


@pytest.mark.parametrize("mutation", ["rand_1", "best_1", "current_to_best_1"])
def test_reproducibility_for_each_supported_strategy(mutation: str) -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation=mutation, F=0.5, CR=0.9)
    run_1 = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    run_2 = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))

    assert np.allclose(run_1.best_solution, run_2.best_solution)
    assert np.isclose(run_1.best_value, run_2.best_value)
    assert np.allclose(run_1.best_value_history, run_2.best_value_history)


def test_random_de_strategy_can_change_between_generations() -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))

    transition_history = result.strategy_history[1:]
    assert len(transition_history) == cfg.iters - 1
    assert set(transition_history).issubset({"rand_1", "best_1", "current_to_best_1"})
    assert len(set(transition_history)) >= 2


def test_random_de_selector_can_access_all_three_strategies() -> None:
    def cycle_selector(transition_index: int, available: list[str] | tuple[str, ...]) -> str:
        return available[(transition_index - 1) % len(available)]

    cfg = DEConfig(seed=7, iters=4, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    result = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=False),
        strategy_selector=cycle_selector,
    )
    transition_history = result.strategy_history[1:]
    assert transition_history == ["rand_1", "best_1", "current_to_best_1"]


def test_random_de_strategy_history_reproducible_with_same_seed() -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    run_1 = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    run_2 = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))

    assert run_1.strategy_history == run_2.strategy_history
    assert np.allclose(run_1.best_solution, run_2.best_solution)
    assert np.isclose(run_1.best_value, run_2.best_value)
    assert np.allclose(run_1.best_value_history, run_2.best_value_history)


def test_random_de_strategy_random_seed_changes_sequence_reproducibly() -> None:
    cfg_a = DEConfig(seed=42, strategy_random_seed=100, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    cfg_b = DEConfig(seed=42, strategy_random_seed=200, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    cfg_a_repeat = DEConfig(seed=42, strategy_random_seed=100, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)

    run_a = run_differential_evolution(sphere, cfg_a, DEDebugConfig(enabled=False))
    run_b = run_differential_evolution(sphere, cfg_b, DEDebugConfig(enabled=False))
    run_a_repeat = run_differential_evolution(sphere, cfg_a_repeat, DEDebugConfig(enabled=False))

    assert run_a.strategy_history == run_a_repeat.strategy_history
    assert run_a.strategy_history != run_b.strategy_history


def test_random_de_keeps_population_state_and_matches_fixed_when_selector_constant() -> None:
    def always_best_selector(transition_index: int, available: list[str] | tuple[str, ...]) -> str:
        del transition_index, available
        return "best_1"

    cfg_random = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    cfg_fixed = DEConfig(seed=42, iters=10, pop_size=10, mutation="best_1", F=0.5, CR=0.9)

    random_as_best = run_differential_evolution(
        sphere,
        cfg_random,
        DEDebugConfig(enabled=False),
        strategy_selector=always_best_selector,
    )
    fixed_best = run_differential_evolution(sphere, cfg_fixed, DEDebugConfig(enabled=False))

    assert np.allclose(random_as_best.best_solution, fixed_best.best_solution)
    assert np.isclose(random_as_best.best_value, fixed_best.best_value)
    assert np.allclose(random_as_best.best_value_history, fixed_best.best_value_history)
    assert random_as_best.function_evaluations == fixed_best.function_evaluations


def test_random_de_function_evaluations_count_is_correct() -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    result = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    assert result.function_evaluations == cfg.pop_size * cfg.iters


def test_random_de_normal_and_debug_results_match() -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    normal = run_differential_evolution(sphere, cfg, DEDebugConfig(enabled=False))
    debug = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )

    assert normal.strategy_history == debug.strategy_history
    assert np.allclose(normal.best_solution, debug.best_solution)
    assert np.isclose(normal.best_value, debug.best_value)
    assert np.allclose(normal.best_value_history, debug.best_value_history)


def test_random_de_debug_trace_matches_chosen_strategy() -> None:
    debug_cfg = DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0)
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9)
    result = run_differential_evolution(sphere, cfg, debug_cfg)

    assert result.debug_trace is not None
    trace = result.debug_trace["mutation_crossover_selection"]
    chosen_strategy = result.strategy_history[debug_cfg.generation_to_trace]
    assert trace["mutation_strategy"] == chosen_strategy
    assert trace["consistency_checks"]["rand_indices_unique"]
    assert trace["consistency_checks"]["rand_indices_in_range"]
    assert trace["consistency_checks"]["mutant_matches_formula"]


@pytest.mark.parametrize("mutation", ["rand_1", "best_1", "current_to_best_1"])
def test_debug_selection_reason_for_maximization_fixed_strategies(mutation: str) -> None:
    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation=mutation, F=0.5, CR=0.9, minimization=False)
    result = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
    )
    assert result.debug_trace is not None
    reason = result.debug_trace["mutation_crossover_selection"]["selection"]["reason"]
    assert "maximization" in reason


def test_debug_selection_reason_for_maximization_random_de() -> None:
    def always_rand_selector(transition_index: int, available: list[str] | tuple[str, ...]) -> str:
        del transition_index, available
        return "rand_1"

    cfg = DEConfig(seed=42, iters=10, pop_size=10, mutation="random_de", F=0.5, CR=0.9, minimization=False)
    result = run_differential_evolution(
        sphere,
        cfg,
        DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0),
        strategy_selector=always_rand_selector,
    )
    assert result.debug_trace is not None
    reason = result.debug_trace["mutation_crossover_selection"]["selection"]["reason"]
    assert "maximization" in reason


def test_uniform_random_strategy_selector_is_statistically_balanced() -> None:
    check_random_state(12345)
    available = ("rand_1", "best_1", "current_to_best_1")
    draws = 9000
    counts = {k: 0 for k in available}
    for i in range(draws):
        picked = uniform_random_strategy_selector(i + 1, available)
        counts[picked] += 1

    expected = draws / len(available)
    chi_square = sum(((counts[k] - expected) ** 2) / expected for k in available)
    # df=2, критическое значение для p=0.01 ~ 9.21.
    assert chi_square < 9.21
