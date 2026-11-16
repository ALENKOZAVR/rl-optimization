from __future__ import annotations

import numpy as np

from rl_optimization.benchmarks import ackley
from rl_optimization.benchmarks import rastrigin
from rl_optimization.benchmarks import rosenbrock
from rl_optimization.benchmarks import schwefel
from rl_optimization.benchmarks import sphere


def test_sphere_at_zero_is_zero() -> None:
    x = np.zeros((1, 5), dtype=np.float64)
    assert np.isclose(sphere(x)[0], 0.0)


def test_ackley_at_zero_is_zero() -> None:
    x = np.zeros((1, 5), dtype=np.float64)
    assert np.isclose(ackley(x)[0], 0.0, atol=1e-12)


def test_rastrigin_at_zero_is_zero() -> None:
    x = np.zeros((1, 5), dtype=np.float64)
    assert np.isclose(rastrigin(x)[0], 0.0, atol=1e-12)


def test_rosenbrock_at_ones_is_zero() -> None:
    x = np.ones((1, 5), dtype=np.float64)
    assert np.isclose(rosenbrock(x)[0], 0.0, atol=1e-12)


def test_schwefel_at_standard_optimum_is_near_zero() -> None:
    optimum = 420.968746
    x = np.full((1, 5), optimum, dtype=np.float64)
    assert np.isclose(schwefel(x)[0], 0.0, atol=1e-3)
