from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from .agents.q_learning import QLearningAgent

if TYPE_CHECKING:
    from .algorithms.differential_evolution import DEConfig
    from .algorithms.differential_evolution import DEDebugConfig
    from .algorithms.differential_evolution import DEResult
    from .algorithms.differential_evolution import RLConfig
    from .algorithms.differential_evolution import run_differential_evolution
    from .algorithms.differential_evolution import run_rl_differential_evolution

__all__ = [
    "QLearningAgent",
    "DEConfig",
    "DEDebugConfig",
    "DEResult",
    "RLConfig",
    "run_differential_evolution",
    "run_rl_differential_evolution",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DEConfig",
        "DEDebugConfig",
        "DEResult",
        "RLConfig",
        "run_differential_evolution",
        "run_rl_differential_evolution",
    }:
        from .algorithms.differential_evolution import DEConfig
        from .algorithms.differential_evolution import DEDebugConfig
        from .algorithms.differential_evolution import DEResult
        from .algorithms.differential_evolution import RLConfig
        from .algorithms.differential_evolution import run_differential_evolution
        from .algorithms.differential_evolution import run_rl_differential_evolution

        mapping = {
            "DEConfig": DEConfig,
            "DEDebugConfig": DEDebugConfig,
            "DEResult": DEResult,
            "RLConfig": RLConfig,
            "run_differential_evolution": run_differential_evolution,
            "run_rl_differential_evolution": run_rl_differential_evolution,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
