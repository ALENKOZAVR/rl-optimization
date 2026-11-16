from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sphere(population: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sum(population**2, axis=1, dtype=np.float64)


def ackley(population: NDArray[np.float64]) -> NDArray[np.float64]:
    n = population.shape[1]
    sum_sq = np.sum(population**2, axis=1, dtype=np.float64)
    sum_cos = np.sum(np.cos(2.0 * np.pi * population), axis=1, dtype=np.float64)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + 20.0 + np.e


def rastrigin(population: NDArray[np.float64]) -> NDArray[np.float64]:
    n = population.shape[1]
    return 10.0 * n + np.sum(population**2 - 10.0 * np.cos(2.0 * np.pi * population), axis=1)


def rosenbrock(population: NDArray[np.float64]) -> NDArray[np.float64]:
    x = population[:, :-1]
    y = population[:, 1:]
    return np.sum(100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2, axis=1, dtype=np.float64)


def schwefel(population: NDArray[np.float64]) -> NDArray[np.float64]:
    n = population.shape[1]
    abs_x = np.abs(population)
    return 418.9829 * n - np.sum(population * np.sin(np.sqrt(abs_x)), axis=1, dtype=np.float64)


BENCHMARK_FUNCTIONS = {
    "sphere": sphere,
    "ackley": ackley,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "schwefel": schwefel,
}
