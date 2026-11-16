from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


DEFAULT_STATES: tuple[str, str, str] = ("early", "middle", "late")
DEFAULT_ACTIONS: tuple[str, str, str] = ("rand_1", "best_1", "current_to_best_1")


@dataclass(frozen=True)
class QUpdateTrace:
    state: str
    action: str
    reward: float
    next_state: str | None
    done: bool
    old_q: float
    max_next_q: float
    td_error: float
    new_q: float
    terminal_transition: bool
    formula: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "old_q": self.old_q,
            "max_next_q": self.max_next_q,
            "td_error": self.td_error,
            "new_q": self.new_q,
            "terminal_transition": self.terminal_transition,
            "formula": self.formula,
        }


class QLearningAgent:
    def __init__(
        self,
        states: Sequence[str] = DEFAULT_STATES,
        actions: Sequence[str] = DEFAULT_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int = 42,
    ) -> None:
        self._states = self._validate_labels(states, "states")
        self._actions = self._validate_labels(actions, "actions")
        self._state_to_idx = {state: idx for idx, state in enumerate(self._states)}
        self._action_to_idx = {action: idx for idx, action in enumerate(self._actions)}
        self._validate_probability(alpha, "alpha")
        self._validate_probability(gamma, "gamma")
        self._validate_probability(epsilon, "epsilon")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

        self._q_table: NDArray[np.float64] = np.zeros(
            (len(self._states), len(self._actions)), dtype=np.float64
        )
        self.update_count = 0

    @property
    def states(self) -> tuple[str, ...]:
        return self._states

    @property
    def actions(self) -> tuple[str, ...]:
        return self._actions

    def set_epsilon(self, epsilon: float) -> None:
        self._validate_probability(epsilon, "epsilon")
        self.epsilon = float(epsilon)

    def get_q_table(self) -> NDArray[np.float64]:
        return self._q_table.copy()

    def reset(self, reset_rng: bool = False) -> None:
        self._q_table.fill(0.0)
        self.update_count = 0
        if reset_rng:
            self._rng = np.random.RandomState(self.seed)

    def select_action(self, state: str) -> str:
        s_idx = self._state_index(state)
        if self._rng.rand() < self.epsilon:
            action_idx = int(self._rng.randint(0, len(self._actions)))
            return self._actions[action_idx]

        state_q = self._q_table[s_idx]
        max_q = float(np.max(state_q))
        best_indices = np.flatnonzero(np.isclose(state_q, max_q))
        action_idx = int(best_indices[int(self._rng.randint(0, len(best_indices)))])
        return self._actions[action_idx]

    def get_preferred_action(self, state: str) -> str:
        s_idx = self._state_index(state)
        state_q = self._q_table[s_idx]
        max_q = float(np.max(state_q))
        best_indices = np.flatnonzero(np.isclose(state_q, max_q))
        action_idx = int(best_indices[int(self._rng.randint(0, len(best_indices)))])
        return self._actions[action_idx]

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str | None,
        done: bool = False,
        debug: bool = False,
    ) -> dict[str, Any] | None:
        s_idx = self._state_index(state)
        a_idx = self._action_index(action)
        old_q = float(self._q_table[s_idx, a_idx])
        reward_value = float(reward)

        if done:
            max_next_q = 0.0
            td_error = reward_value - old_q
            formula = "Q(s,a) <- Q(s,a) + alpha * (r - Q(s,a))"
        else:
            if next_state is None:
                raise ValueError("next_state must be provided when done=False")
            ns_idx = self._state_index(next_state)
            max_next_q = float(np.max(self._q_table[ns_idx]))
            td_error = reward_value + self.gamma * max_next_q - old_q
            formula = "Q(s,a) <- Q(s,a) + alpha * (r + gamma * maxQ(s',.) - Q(s,a))"

        new_q = old_q + self.alpha * td_error
        self._q_table[s_idx, a_idx] = new_q
        self.update_count += 1

        if not debug:
            return None

        trace = QUpdateTrace(
            state=state,
            action=action,
            reward=reward_value,
            next_state=next_state,
            done=done,
            old_q=old_q,
            max_next_q=max_next_q,
            td_error=float(td_error),
            new_q=float(new_q),
            terminal_transition=done,
            formula=formula,
        )
        return trace.to_dict()

    def _state_index(self, state: str) -> int:
        if state not in self._state_to_idx:
            raise ValueError(f"Unknown state '{state}'. Known states: {self._states}")
        return self._state_to_idx[state]

    def _action_index(self, action: str) -> int:
        if action not in self._action_to_idx:
            raise ValueError(f"Unknown action '{action}'. Known actions: {self._actions}")
        return self._action_to_idx[action]

    @staticmethod
    def _validate_labels(labels: Sequence[str], label_type: str) -> tuple[str, ...]:
        values = tuple(str(x) for x in labels)
        if len(values) == 0:
            raise ValueError(f"{label_type} must not be empty")
        if any(v.strip() == "" for v in values):
            raise ValueError(f"{label_type} must not contain empty names")
        if len(set(values)) != len(values):
            raise ValueError(f"{label_type} must be unique")
        return values

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")
