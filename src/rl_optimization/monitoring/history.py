from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def save_history_csv(best_value_history: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "best_value", "delta_to_previous"])
        writer.writeheader()
        previous: float | None = None
        for generation, value in enumerate(best_value_history):
            delta = "" if previous is None else value - previous
            writer.writerow(
                {
                    "generation": generation,
                    "best_value": value,
                    "delta_to_previous": delta,
                }
            )
            previous = value


def save_result_json(result: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = dict(result)
    if "best_solution" in serializable:
        serializable["best_solution"] = list(serializable["best_solution"])

    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def save_simple_plot(best_value_history: list[float], path: str | Path, title: str = "DE on Sphere") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(range(len(best_value_history)), best_value_history, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best objective value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
