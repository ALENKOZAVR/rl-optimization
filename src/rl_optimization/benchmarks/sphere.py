from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sphere(population: NDArray[np.float64]) -> NDArray[np.float64]:
    """Векторизованная функция Sphere для массива особей формы (N, D)."""
    return np.sum(population**2, axis=1, dtype=np.float64)
