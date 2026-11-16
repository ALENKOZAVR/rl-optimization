from __future__ import annotations

import numpy as np

from rl_optimization.agents.q_learning import QLearningAgent
from rl_optimization.algorithms.differential_evolution import budget_state_from_calls
from rl_optimization.algorithms.differential_evolution import DEConfig
from rl_optimization.algorithms.differential_evolution import DEDebugConfig
from rl_optimization.algorithms.differential_evolution import RLConfig
from rl_optimization.algorithms.differential_evolution import run_rl_differential_evolution
from rl_optimization.algorithms.differential_evolution import strict_improvement_reward_from_internal_fitness
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
