from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys

from rl_optimization.agents.q_learning import QLearningAgent


def test_q_table_shape_and_initial_values() -> None:
    agent = QLearningAgent(seed=42)
    q = agent.get_q_table()
    assert q.shape == (3, 3)
    assert np.allclose(q, np.zeros((3, 3), dtype=np.float64))


def test_q_learning_update_formula_non_terminal() -> None:
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0, seed=42)
    agent.update("middle", "best_1", reward=2.0, next_state="late", done=False)
    # old_q=0, max_next_q=0 => new_q = 0 + 0.1*(2 + 0 - 0) = 0.2
    assert np.isclose(agent.get_q_table()[1, 1], 0.2)


def test_q_learning_update_formula_terminal() -> None:
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.0, seed=42)
    agent.update("late", "current_to_best_1", reward=1.0, next_state=None, done=True)
    # old_q=0 => new_q = 0 + 0.1*(1 - 0) = 0.1
    assert np.isclose(agent.get_q_table()[2, 2], 0.1)


def test_epsilon_greedy_with_zero_epsilon_selects_greedy_action() -> None:
    agent = QLearningAgent(epsilon=0.0, seed=42)
    agent.update("early", "rand_1", reward=1.0, next_state="middle", done=False)
    selected = [agent.select_action("early") for _ in range(10)]
    assert all(action == "rand_1" for action in selected)


def test_epsilon_greedy_with_one_epsilon_explores_actions() -> None:
    agent = QLearningAgent(epsilon=1.0, seed=42)
    selected = [agent.select_action("middle") for _ in range(200)]
    assert set(selected) == {"rand_1", "best_1", "current_to_best_1"}


def test_tie_breaking_among_equal_q_values_uses_rng() -> None:
    agent_1 = QLearningAgent(epsilon=0.0, seed=7)
    agent_2 = QLearningAgent(epsilon=0.0, seed=7)
    picks_1 = [agent_1.select_action("early") for _ in range(20)]
    picks_2 = [agent_2.select_action("early") for _ in range(20)]
    assert picks_1 == picks_2
    assert set(picks_1).issubset({"rand_1", "best_1", "current_to_best_1"})
    assert len(set(picks_1)) > 1


def test_reproducibility_with_same_seed() -> None:
    agent_1 = QLearningAgent(epsilon=0.3, seed=123)
    agent_2 = QLearningAgent(epsilon=0.3, seed=123)
    for _ in range(30):
        a1 = agent_1.select_action("middle")
        a2 = agent_2.select_action("middle")
        assert a1 == a2
        agent_1.update("middle", a1, reward=1.0 if a1 == "best_1" else 0.0, next_state="late", done=False)
        agent_2.update("middle", a2, reward=1.0 if a2 == "best_1" else 0.0, next_state="late", done=False)
    assert np.allclose(agent_1.get_q_table(), agent_2.get_q_table())


def test_select_action_does_not_change_q_table() -> None:
    agent = QLearningAgent(seed=42)
    before = agent.get_q_table()
    _ = [agent.select_action("early") for _ in range(20)]
    after = agent.get_q_table()
    assert np.allclose(before, after)


def test_reset_clears_q_table_and_update_count() -> None:
    agent = QLearningAgent(seed=42)
    agent.update("early", "rand_1", reward=1.0, next_state="middle", done=False)
    assert agent.update_count == 1
    agent.reset()
    assert agent.update_count == 0
    assert np.allclose(agent.get_q_table(), 0.0)


def test_reset_with_rng_rewind_reproduces_action_sequence() -> None:
    agent = QLearningAgent(epsilon=1.0, seed=42)
    seq_before = [agent.select_action("early") for _ in range(8)]
    agent.update("early", "rand_1", reward=1.0, next_state="middle", done=False)
    agent.reset(reset_rng=True)
    seq_after = [agent.select_action("early") for _ in range(8)]
    assert seq_before == seq_after
    assert np.allclose(agent.get_q_table(), 0.0)
    assert agent.update_count == 0


def test_root_import_allows_qlearning_without_importing_de_dependencies() -> None:
    script = """
import builtins
_orig_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'thefittest' or name.startswith('thefittest.'):
        raise ImportError('blocked for test')
    return _orig_import(name, *args, **kwargs)
builtins.__import__ = blocked
import rl_optimization
_ = rl_optimization.QLearningAgent
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_invalid_parameters_and_unknown_state_action_raise_error() -> None:
    with pytest.raises(ValueError):
        QLearningAgent(alpha=-0.1)
    with pytest.raises(ValueError):
        QLearningAgent(gamma=1.2)
    with pytest.raises(ValueError):
        QLearningAgent(epsilon=-0.2)
    with pytest.raises(ValueError):
        QLearningAgent(states=())
    with pytest.raises(ValueError):
        QLearningAgent(actions=("rand_1", "rand_1"))

    agent = QLearningAgent(seed=42)
    with pytest.raises(ValueError):
        agent.select_action("unknown_state")
    with pytest.raises(ValueError):
        agent.update("early", "unknown_action", 1.0, "middle", done=False)
    with pytest.raises(ValueError):
        agent.update("early", "rand_1", 1.0, None, done=False)


def test_normal_and_debug_update_have_same_math() -> None:
    agent_normal = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)
    agent_debug = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1, seed=42)

    agent_normal.update("early", "rand_1", reward=1.0, next_state="middle", done=False, debug=False)
    trace = agent_debug.update("early", "rand_1", reward=1.0, next_state="middle", done=False, debug=True)

    assert trace is not None
    assert np.allclose(agent_normal.get_q_table(), agent_debug.get_q_table())
    assert np.isclose(trace["old_q"], 0.0)
    assert np.isclose(trace["max_next_q"], 0.0)
    assert np.isclose(trace["new_q"], 0.1)
    assert not trace["terminal_transition"]


def test_learning_on_artificial_environment_prefers_expected_actions() -> None:
    agent = QLearningAgent(alpha=0.3, gamma=0.9, epsilon=0.2, seed=42)
    states = ("early", "middle", "late")
    optimal = {
        "early": "rand_1",
        "middle": "best_1",
        "late": "current_to_best_1",
    }

    for _ in range(200):
        for i, state in enumerate(states):
            action = agent.select_action(state)
            reward = 1.0 if action == optimal[state] else 0.0
            next_state = states[i + 1] if i < len(states) - 1 else None
            done = i == len(states) - 1
            agent.update(state, action, reward=reward, next_state=next_state, done=done)

    assert agent.get_preferred_action("early") == "rand_1"
    assert agent.get_preferred_action("middle") == "best_1"
    assert agent.get_preferred_action("late") == "current_to_best_1"
    q = agent.get_q_table()
    assert q[0, 0] > q[0, 1] and q[0, 0] > q[0, 2]
    assert q[1, 1] > q[1, 0] and q[1, 1] > q[1, 2]
    assert q[2, 2] > q[2, 0] and q[2, 2] > q[2, 1]
