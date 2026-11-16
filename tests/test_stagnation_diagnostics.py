from __future__ import annotations

import numpy as np

from rl_optimization.experiments.stagnation_diagnostics import compute_non_improvement_run_lengths
from rl_optimization.experiments.stagnation_diagnostics import recompute_stagnation_from_best_history
from rl_optimization.experiments.stagnation_diagnostics import relative_improvement


def test_relative_improvement_uses_function_scale_near_zero() -> None:
    before = 1e-12
    after = 0.5e-12
    rel = relative_improvement(before, after, function_scale=10.0)
    assert np.isclose(rel, (before - after) / 10.0)


def test_recompute_stagnation_counter_and_flags_minimization() -> None:
    history = [10.0, 9.0, 9.0, 9.0, 8.5, 8.5]
    out = recompute_stagnation_from_best_history(
        history,
        minimization=True,
        stagnation_threshold=2,
    )
    assert out["strict_improvements"] == [True, False, False, True, False]
    assert out["counters_before"] == [0, 0, 1, 2, 0]
    assert out["counters_after"] == [0, 1, 2, 0, 1]
    assert out["flags_before"] == ["improving", "improving", "improving", "stagnating", "improving"]


def test_compute_non_improvement_run_lengths() -> None:
    strict = [True, False, False, True, False, False, False, True]
    lengths = compute_non_improvement_run_lengths(strict)
    assert lengths == [2, 3]
