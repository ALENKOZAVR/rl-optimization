from __future__ import annotations

import numpy as np

from rl_optimization.agents.q_learning import QLearningAgent
from rl_optimization.algorithms.differential_evolution import budget_state_from_calls
from rl_optimization.algorithms.differential_evolution import DEConfig
from rl_optimization.algorithms.differential_evolution import DEDebugConfig
from rl_optimization.algorithms.differential_evolution import encode_rl_state
from rl_optimization.algorithms.differential_evolution import RL_STATE_DEFINITION_BUDGET_THIRDS
from rl_optimization.algorithms.differential_evolution import RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION
from rl_optimization.algorithms.differential_evolution import rl_states_for_definition
from rl_optimization.algorithms.differential_evolution import RLConfig
from rl_optimization.algorithms.differential_evolution import run_rl_differential_evolution
from rl_optimization.algorithms.differential_evolution import stagnation_flag_from_counter
from rl_optimization.algorithms.differential_evolution import strict_improvement_reward_from_internal_fitness
from rl_optimization.algorithms.differential_evolution import update_stagnation_counter
from rl_optimization.benchmarks.sphere import sphere


def test_budget_state_boundaries() -> None:
    total = 99
    assert budget_state_from_calls(0, total) == "early"
    assert budget_state_from_calls(32, total) == "early"
    assert budget_state_from_calls(33, total) == "middle"
    assert budget_state_from_calls(65, total) == "middle"
    assert budget_state_from_calls(66, total) == "late"
    assert budget_state_from_calls(99, total) == "late"


def test_reward_from_strict_internal_improvements() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    trial = np.array([1.0, 2.1, 2.9, 4.0], dtype=np.float64)
    reward, successes = strict_improvement_reward_from_internal_fitness(target, trial)
    assert successes == 1
    assert np.isclose(reward, 0.25)


def test_rl_de_does_not_add_extra_objective_evaluations() -> None:
    counter = {"calls": 0}

    def counted_sphere(population: np.ndarray) -> np.ndarray:
        counter["calls"] += int(len(population))
        return np.sum(population**2, axis=1, dtype=np.float64)

    cfg = DEConfig(iters=10, pop_size=10, seed=42, objective_mode="population")
    rl_cfg = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    result = run_rl_differential_evolution(counted_sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))

    assert result.function_evaluations == cfg.pop_size * cfg.iters
    assert counter["calls"] == result.function_evaluations


def _prepare_deterministic_agent() -> QLearningAgent:
    agent = QLearningAgent(epsilon=0.0, seed=42)
    # Поднимаем Q для желаемого действия в каждом состоянии.
    agent.update("early", "rand_1", reward=1.0, next_state=None, done=True)
    agent.update("middle", "best_1", reward=1.0, next_state=None, done=True)
    agent.update("late", "current_to_best_1", reward=1.0, next_state=None, done=True)
    return agent


def test_rl_de_uses_agent_selected_strategy_for_generation() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.0, seed=42, carry_learning=True)
    agent = _prepare_deterministic_agent()
    result = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False), agent=agent)

    expected_transitions = ["rand_1"] * 3 + ["best_1"] * 3 + ["current_to_best_1"] * 3
    assert result.strategy_history[1:] == expected_transitions


def test_rl_de_updates_q_table_and_terminal_update() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    debug_cfg = DEDebugConfig(enabled=True, generation_to_trace=9, individual_index_to_trace=0)
    result = run_rl_differential_evolution(sphere, cfg, rl_cfg, debug_cfg)

    assert result.final_q_table is not None
    assert np.any(np.asarray(result.final_q_table) != 0.0)
    assert result.rl_history is not None
    assert result.rl_history[-1]["done"] is True
    assert result.debug_trace is not None
    q_trace = result.debug_trace["q_update_trace"]
    assert q_trace is not None
    assert q_trace["terminal_transition"]


def test_rl_de_reproducibility_with_same_seed() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    r1 = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    r2 = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))

    assert np.allclose(r1.best_solution, r2.best_solution)
    assert np.isclose(r1.best_value, r2.best_value)
    assert np.allclose(r1.best_value_history, r2.best_value_history)
    assert r1.strategy_history == r2.strategy_history
    assert r1.final_q_table == r2.final_q_table


def test_rl_de_normal_and_debug_match() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    normal = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    debug = run_rl_differential_evolution(
        sphere, cfg, rl_cfg, DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0)
    )

    assert np.allclose(normal.best_solution, debug.best_solution)
    assert np.isclose(normal.best_value, debug.best_value)
    assert np.allclose(normal.best_value_history, debug.best_value_history)
    assert normal.strategy_history == debug.strategy_history
    assert normal.final_q_table == debug.final_q_table


def test_rl_de_updates_agent_exactly_once_per_generation_transition() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42, carry_learning=True)
    agent = QLearningAgent(seed=42)
    _ = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False), agent=agent)
    assert agent.update_count == cfg.iters - 1


def test_stagnation_counter_starts_with_zero_before_first_action() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
        stagnation_threshold=5,
    )
    result = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    assert result.rl_history is not None
    assert result.rl_history[0]["stagnation_counter_before"] == 0


def test_update_stagnation_counter_reset_on_strict_improvement() -> None:
    assert update_stagnation_counter(4, best_value_before=10.0, best_value_after=9.0, minimization=True) == 0
    assert update_stagnation_counter(3, best_value_before=10.0, best_value_after=11.0, minimization=False) == 0


def test_update_stagnation_counter_increments_without_improvement() -> None:
    assert update_stagnation_counter(2, best_value_before=10.0, best_value_after=10.0, minimization=True) == 3
    assert update_stagnation_counter(2, best_value_before=10.0, best_value_after=11.0, minimization=True) == 3


def test_equal_best_value_is_not_strict_improvement() -> None:
    assert update_stagnation_counter(0, best_value_before=1.0, best_value_after=1.0, minimization=True) == 1
    assert update_stagnation_counter(0, best_value_before=1.0, best_value_after=1.0, minimization=False) == 1


def test_stagnation_threshold_boundary_flag() -> None:
    assert stagnation_flag_from_counter(4, 5) == "improving"
    assert stagnation_flag_from_counter(5, 5) == "stagnating"


def test_stagnation_six_states_mapping() -> None:
    states = rl_states_for_definition(RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION)
    assert len(states) == 6
    assert states == (
        "early_improving",
        "early_stagnating",
        "middle_improving",
        "middle_stagnating",
        "late_improving",
        "late_stagnating",
    )


def test_encode_rl_state_for_stagnation_definition() -> None:
    total = 300
    assert encode_rl_state(100, total, RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION, 0, 5) == "middle_improving"
    assert encode_rl_state(100, total, RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION, 5, 5) == "middle_stagnating"


def test_transition_from_improving_to_stagnating_and_back() -> None:
    counter = 0
    # Five consecutive non-improving transitions.
    for _ in range(5):
        counter = update_stagnation_counter(counter, 1.0, 1.0, minimization=True)
    assert stagnation_flag_from_counter(counter, 5) == "stagnating"
    # Strict improvement resets.
    counter = update_stagnation_counter(counter, 1.0, 0.5, minimization=True)
    assert counter == 0
    assert stagnation_flag_from_counter(counter, 5) == "improving"


def test_v2_state_before_action_uses_only_pre_generation_information() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
        stagnation_threshold=5,
    )
    result = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    assert result.rl_history is not None

    for step in result.rl_history:
        transition = int(step["generation_transition"])
        used_calls_before = cfg.pop_size * transition
        expected_state = encode_rl_state(
            used_calls=used_calls_before,
            total_budget_calls=cfg.pop_size * cfg.iters,
            state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
            stagnation_counter=int(step["stagnation_counter_before"]),
            stagnation_threshold=5,
        )
        assert step["state"] == expected_state


def test_v2_does_not_add_extra_objective_evaluations() -> None:
    counter = {"calls": 0}

    def counted_sphere(population: np.ndarray) -> np.ndarray:
        counter["calls"] += int(len(population))
        return np.sum(population**2, axis=1, dtype=np.float64)

    cfg = DEConfig(iters=10, pop_size=10, seed=42, objective_mode="population")
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
        stagnation_threshold=5,
    )
    result = run_rl_differential_evolution(counted_sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))

    assert result.function_evaluations == cfg.pop_size * cfg.iters
    assert counter["calls"] == result.function_evaluations


def test_v2_q_table_shape_updates_and_terminal_step() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
        stagnation_threshold=5,
    )
    result = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))

    assert result.final_q_table is not None
    q = np.asarray(result.final_q_table, dtype=np.float64)
    assert q.shape == (6, 3)
    assert np.any(q != 0.0)
    assert result.rl_history is not None
    assert result.rl_history[-1]["done"] is True


def test_v2_reproducibility_with_same_seed() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
        stagnation_threshold=5,
    )
    r1 = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    r2 = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))

    assert np.allclose(r1.best_solution, r2.best_solution)
    assert np.isclose(r1.best_value, r2.best_value)
    assert np.allclose(r1.best_value_history, r2.best_value_history)
    assert r1.strategy_history == r2.strategy_history
    assert r1.final_q_table == r2.final_q_table


def test_v2_normal_and_debug_match() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS_WITH_STAGNATION,
        stagnation_threshold=5,
    )
    normal = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    debug = run_rl_differential_evolution(
        sphere, cfg, rl_cfg, DEDebugConfig(enabled=True, generation_to_trace=1, individual_index_to_trace=0)
    )

    assert np.allclose(normal.best_solution, debug.best_solution)
    assert np.isclose(normal.best_value, debug.best_value)
    assert np.allclose(normal.best_value_history, debug.best_value_history)
    assert normal.strategy_history == debug.strategy_history
    assert normal.final_q_table == debug.final_q_table


def test_v1_remains_budget_state_only() -> None:
    cfg = DEConfig(iters=10, pop_size=10, seed=42)
    rl_cfg = RLConfig(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=42,
        state_definition=RL_STATE_DEFINITION_BUDGET_THIRDS,
    )
    result = run_rl_differential_evolution(sphere, cfg, rl_cfg, DEDebugConfig(enabled=False))
    assert result.rl_history is not None
    assert {str(step["state"]) for step in result.rl_history}.issubset({"early", "middle", "late"})
