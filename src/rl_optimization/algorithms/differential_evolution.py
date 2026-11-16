from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import inspect
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from rl_optimization.agents.q_learning import QLearningAgent
from thefittest.optimizers import DifferentialEvolution
from thefittest.optimizers._differentialevolution import bounds_control
from thefittest.utils.random import flip_coin
from thefittest.utils.random import randint
from thefittest.utils.random import random_sample

SUPPORTED_MUTATION_ALIASES: dict[str, str] = {
    "rand_1": "rand_1",
    "rand/1": "rand_1",
    "rand/1/bin": "rand_1",
    "de/rand/1": "rand_1",
    "de/rand/1/bin": "rand_1",
    "best_1": "best_1",
    "best/1": "best_1",
    "best/1/bin": "best_1",
    "de/best/1": "best_1",
    "de/best/1/bin": "best_1",
    "current_to_best_1": "current_to_best_1",
    "current-to-best/1": "current_to_best_1",
    "current-to-best/1/bin": "current_to_best_1",
    "de/current-to-best/1": "current_to_best_1",
    "de/current-to-best/1/bin": "current_to_best_1",
    "random_de": "random_de",
    "random-de": "random_de",
    "random": "random_de",
    "random/bin": "random_de",
    "de/random": "random_de",
    "de/random/bin": "random_de",
}
RANDOM_DE_STRATEGIES: tuple[str, str, str] = ("rand_1", "best_1", "current_to_best_1")
StrategySelector = Callable[[int, Sequence[str]], str]


@dataclass(frozen=True)
class DEConfig:
    num_variables: int = 2
    left_border: float = -5.12
    right_border: float = 5.12
    pop_size: int = 10
    F: float = 0.5
    CR: float = 0.9
    iters: int = 10
    seed: int = 42
    mutation: str = "rand_1"
    objective_mode: str = "population"
    strategy_random_seed: int | None = None
    elitism: bool = True
    minimization: bool = True


@dataclass(frozen=True)
class DEDebugConfig:
    enabled: bool = False
    generation_to_trace: int = 1
    individual_index_to_trace: int = 0


@dataclass(frozen=True)
class RLConfig:
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    seed: int = 42
    carry_learning: bool = False


@dataclass
class DEResult:
    config: DEConfig
    best_solution: NDArray[np.float64]
    best_value: float
    best_value_history: list[float]
    function_evaluations: int
    population_history: list[NDArray[np.float64]]
    strategy_history: list[str]
    debug_trace: dict[str, Any] | None
    thefittest_version: str
    rl_history: list[dict[str, Any]] | None = None
    final_q_table: list[list[float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "best_solution": self.best_solution.tolist(),
            "best_value": self.best_value,
            "best_value_history": self.best_value_history,
            "function_evaluations": self.function_evaluations,
            "strategy_history": self.strategy_history,
            "rl_history": self.rl_history,
            "final_q_table": self.final_q_table,
            "debug_trace": self.debug_trace,
            "thefittest_version": self.thefittest_version,
        }


def clip_to_bounds(
    vector: NDArray[np.float64], left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    return bounds_control(vector, left, right)


def rand_1_mutation_from_indices(
    population: NDArray[np.float64], r1: int, r2: int, r3: int, F: float
) -> NDArray[np.float64]:
    return population[r3] + np.float64(F) * (population[r1] - population[r2])


def best_1_mutation_from_indices(
    best_vector: NDArray[np.float64], population: NDArray[np.float64], r1: int, r2: int, F: float
) -> NDArray[np.float64]:
    return best_vector + np.float64(F) * (population[r1] - population[r2])


def current_to_best_1_mutation_from_indices(
    current_vector: NDArray[np.float64],
    best_vector: NDArray[np.float64],
    population: NDArray[np.float64],
    r1: int,
    r2: int,
    F: float,
) -> NDArray[np.float64]:
    return (
        current_vector
        + np.float64(F) * (best_vector - current_vector)
        + np.float64(F) * (population[r1] - population[r2])
    )


def normalize_mutation_strategy(mutation: str) -> str:
    key = mutation.strip().lower()
    if key not in SUPPORTED_MUTATION_ALIASES:
        supported = ", ".join(sorted(set(SUPPORTED_MUTATION_ALIASES.keys())))
        raise ValueError(f"Unsupported mutation strategy '{mutation}'. Supported aliases: {supported}")
    return SUPPORTED_MUTATION_ALIASES[key]


def normalize_objective_mode(objective_mode: str) -> str:
    value = objective_mode.strip().lower()
    aliases = {
        "population": "population",
        "vectorized": "population",
        "batch": "population",
        "individual": "individual",
        "single": "individual",
    }
    if value not in aliases:
        raise ValueError(
            f"Unsupported objective mode '{objective_mode}'. Use 'population' or 'individual'."
        )
    return aliases[value]


def uniform_random_strategy_selector(
    transition_index: int,
    available_strategies: Sequence[str],
    rng: np.random.RandomState | None = None,
) -> str:
    del transition_index
    if rng is None:
        idx = int(np.random.randint(0, len(available_strategies)))
    else:
        idx = int(rng.randint(0, len(available_strategies)))
    return str(available_strategies[idx])


def should_select_trial_minimization(target_value: float, trial_value: float) -> bool:
    return trial_value <= target_value


def budget_state_from_calls(used_calls: int, total_budget_calls: int) -> str:
    if total_budget_calls <= 0:
        raise ValueError("total_budget_calls must be > 0")
    ratio = float(used_calls) / float(total_budget_calls)
    if ratio < (1.0 / 3.0):
        return "early"
    if ratio < (2.0 / 3.0):
        return "middle"
    return "late"


def strict_improvement_reward_from_internal_fitness(
    target_internal_fitness: NDArray[np.float64],
    trial_internal_fitness: NDArray[np.float64],
) -> tuple[float, int]:
    if len(target_internal_fitness) != len(trial_internal_fitness):
        raise ValueError("target_internal_fitness and trial_internal_fitness must have equal length")
    strict_improvements = trial_internal_fitness > target_internal_fitness
    successful_replacements = int(np.count_nonzero(strict_improvements))
    reward = successful_replacements / float(len(target_internal_fitness))
    return reward, successful_replacements


class _RandomStrategySwitchingMixin:
    def __init__(
        self,
        *args: Any,
        strategy_selector: StrategySelector | None = None,
        available_strategies: Sequence[str] = RANDOM_DE_STRATEGIES,
        **kwargs: Any,
    ) -> None:
        strategy_seed = kwargs.pop("strategy_random_seed", None)
        random_state = kwargs.get("random_state")
        super().__init__(*args, **kwargs)
        self._strategy_selector = strategy_selector or uniform_random_strategy_selector
        self._available_strategies = tuple(available_strategies)
        self.strategy_history_transitions: list[str] = []
        self._strategy_transition_index = 0
        if strategy_seed is None:
            if isinstance(random_state, (int, np.integer)):
                strategy_seed = int(random_state) + 100_003
            elif isinstance(random_state, np.random.RandomState):
                strategy_seed = int(random_state.get_state()[1][0]) + 100_003
        self._strategy_rng = np.random.RandomState(strategy_seed) if strategy_seed is not None else np.random.RandomState()

    def _choose_and_apply_strategy(self) -> str:
        self._strategy_transition_index += 1
        selector_params = inspect.signature(self._strategy_selector).parameters
        if len(selector_params) >= 3:
            strategy = self._strategy_selector(
                self._strategy_transition_index, self._available_strategies, self._strategy_rng
            )
        else:
            strategy = self._strategy_selector(self._strategy_transition_index, self._available_strategies)
        strategy = normalize_mutation_strategy(strategy)
        if strategy not in self._available_strategies:
            raise ValueError(
                f"Strategy selector returned '{strategy}', expected one of {self._available_strategies}"
            )
        self._specified_mutation = strategy
        self.strategy_history_transitions.append(strategy)
        return strategy


class TracingDifferentialEvolution(DifferentialEvolution):
    def __init__(self, *args: Any, debug_config: DEDebugConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._debug_config = debug_config or DEDebugConfig(enabled=False)
        self.debug_trace: dict[str, Any] | None = None
        self._transition_index = 0
        self.best_value_history: list[float] = []

    def _get_new_individ_g_with_trace(
        self,
        individ_g: NDArray[np.float64],
        individ_index: int,
        population_snapshot: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], dict[str, Any] | None]:
        if self._specified_mutation not in {"rand_1", "best_1", "current_to_best_1"}:
            offspring = super()._get_new_individ_g(individ_g, F=self._F, CR=self._CR)
            return offspring, None

        size = np.int64(len(population_snapshot))
        best_vector = self._thefittest._genotype.copy()
        rand_vectors: dict[str, list[float]] = {}
        rand_indices: list[int] = []
        mutation_expression = ""

        if self._specified_mutation == "rand_1":
            r1, r2, r3 = random_sample(range_size=size, quantity=np.int64(3), replace=False)
            rand_indices = [int(r1), int(r2), int(r3)]
            rand_vectors = {
                "x_r1": population_snapshot[int(r1)].copy().tolist(),
                "x_r2": population_snapshot[int(r2)].copy().tolist(),
                "x_r3": population_snapshot[int(r3)].copy().tolist(),
            }
            mutant = rand_1_mutation_from_indices(
                population_snapshot, int(r1), int(r2), int(r3), self._F
            )
            mutation_expression = "v = x_r3 + F * (x_r1 - x_r2)"
        elif self._specified_mutation == "best_1":
            r1, r2 = random_sample(range_size=size, quantity=np.int64(2), replace=False)
            rand_indices = [int(r1), int(r2)]
            rand_vectors = {
                "x_r1": population_snapshot[int(r1)].copy().tolist(),
                "x_r2": population_snapshot[int(r2)].copy().tolist(),
            }
            mutant = best_1_mutation_from_indices(best_vector, population_snapshot, int(r1), int(r2), self._F)
            mutation_expression = "v = x_best + F * (x_r1 - x_r2)"
        else:
            r1, r2 = random_sample(range_size=size, quantity=np.int64(2), replace=False)
            rand_indices = [int(r1), int(r2)]
            rand_vectors = {
                "x_r1": population_snapshot[int(r1)].copy().tolist(),
                "x_r2": population_snapshot[int(r2)].copy().tolist(),
            }
            mutant = current_to_best_1_mutation_from_indices(
                individ_g, best_vector, population_snapshot, int(r1), int(r2), self._F
            )
            mutation_expression = "v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)"

        j_rand = int(randint(0, len(individ_g), 1)[0])
        crossover_mask = np.zeros(len(individ_g), dtype=bool)
        trial = individ_g.copy()
        for j in range(len(individ_g)):
            take_mutant = bool(flip_coin(np.float64(self._CR)) or j == j_rand)
            crossover_mask[j] = take_mutant
            if take_mutant:
                trial[j] = mutant[j]

        bounded_trial = clip_to_bounds(trial, self._left, self._right)

        should_trace = (
            self._debug_config.enabled
            and self._transition_index == self._debug_config.generation_to_trace
            and individ_index == self._debug_config.individual_index_to_trace
        )
        if not should_trace:
            return bounded_trial, None

        trace = {
            "generation_transition": self._transition_index,
            "individual_index": individ_index,
            "mutation_strategy": self._specified_mutation,
            "population_snapshot_before_mutation": population_snapshot.copy().tolist(),
            "best_vector_before_mutation": best_vector.tolist(),
            "target_vector_before": individ_g.copy().tolist(),
            "rand_indices": rand_indices,
            "rand_vectors": rand_vectors,
            "F": float(self._F),
            "mutation_expression": mutation_expression,
            "mutant_vector": mutant.copy().tolist(),
            "CR": float(self._CR),
            "crossover_mask": crossover_mask.tolist(),
            "trial_vector_before_bounds": trial.copy().tolist(),
            "trial_vector_after_bounds": bounded_trial.copy().tolist(),
        }
        return bounded_trial, trace

    def _get_new_population(self) -> None:
        self._transition_index += 1
        target_fitness_before = self._fitness_i.copy()
        population_before_mutation = self._population_g_i.copy()

        trial_population = np.empty_like(self._population_g_i, dtype=np.float64)
        local_trace: dict[str, Any] | None = None

        for i in range(self._pop_size):
            trial, trace = self._get_new_individ_g_with_trace(
                population_before_mutation[i], i, population_before_mutation
            )
            trial_population[i] = trial
            if trace is not None:
                local_trace = trace

        trial_population_ph = self._get_phenotype(trial_population)
        trial_fitness = self._get_fitness(trial_population_ph)
        mask = trial_fitness >= self._fitness_i

        self._population_g_i[mask] = trial_population[mask]
        self._population_ph_i[mask] = trial_population_ph[mask]
        self._fitness_i[mask] = trial_fitness[mask]

        if local_trace is not None:
            idx = int(local_trace["individual_index"])
            target_internal = float(target_fitness_before[idx])
            trial_internal = float(trial_fitness[idx])
            selected_trial = bool(trial_internal >= target_internal)

            snapshot = np.asarray(local_trace["population_snapshot_before_mutation"], dtype=np.float64)
            best_vector = np.asarray(local_trace["best_vector_before_mutation"], dtype=np.float64)
            target_vector = np.asarray(local_trace["target_vector_before"], dtype=np.float64)
            rand_indices = [int(i) for i in local_trace["rand_indices"]]

            if local_trace["mutation_strategy"] == "rand_1":
                r1, r2, r3 = rand_indices
                x_r1 = snapshot[r1]
                x_r2 = snapshot[r2]
                x_r3 = snapshot[r3]
                expected_mutant = x_r3 + float(self._F) * (x_r1 - x_r2)
                expected_rand_keys = {"x_r1", "x_r2", "x_r3"}
                vectors_match = (
                    np.allclose(x_r1, np.asarray(local_trace["rand_vectors"]["x_r1"], dtype=np.float64))
                    and np.allclose(x_r2, np.asarray(local_trace["rand_vectors"]["x_r2"], dtype=np.float64))
                    and np.allclose(x_r3, np.asarray(local_trace["rand_vectors"]["x_r3"], dtype=np.float64))
                )
            elif local_trace["mutation_strategy"] == "best_1":
                r1, r2 = rand_indices
                x_r1 = snapshot[r1]
                x_r2 = snapshot[r2]
                expected_mutant = best_vector + float(self._F) * (x_r1 - x_r2)
                expected_rand_keys = {"x_r1", "x_r2"}
                vectors_match = (
                    np.allclose(x_r1, np.asarray(local_trace["rand_vectors"]["x_r1"], dtype=np.float64))
                    and np.allclose(x_r2, np.asarray(local_trace["rand_vectors"]["x_r2"], dtype=np.float64))
                )
            else:
                r1, r2 = rand_indices
                x_r1 = snapshot[r1]
                x_r2 = snapshot[r2]
                expected_mutant = (
                    target_vector
                    + float(self._F) * (best_vector - target_vector)
                    + float(self._F) * (x_r1 - x_r2)
                )
                expected_rand_keys = {"x_r1", "x_r2"}
                vectors_match = (
                    np.allclose(x_r1, np.asarray(local_trace["rand_vectors"]["x_r1"], dtype=np.float64))
                    and np.allclose(x_r2, np.asarray(local_trace["rand_vectors"]["x_r2"], dtype=np.float64))
                )

            local_trace["selection"] = {
                "target_value": float(self._sign * target_internal),
                "trial_value": float(self._sign * trial_internal),
                "selected": selected_trial,
                "reason": (
                    "trial_value <= target_value (minimization)"
                    if selected_trial
                    else "trial_value > target_value (minimization)"
                )
                if self._sign == -1
                else (
                    "trial_value >= target_value (maximization)"
                    if selected_trial
                    else "trial_value < target_value (maximization)"
                ),
            }
            local_trace["consistency_checks"] = {
                "rand_indices_unique": len(set(rand_indices)) == len(rand_indices),
                "rand_indices_in_range": all(0 <= i < len(snapshot) for i in rand_indices),
                "rand_vector_keys_match_strategy": set(local_trace["rand_vectors"].keys()) == expected_rand_keys,
                "target_index_present_in_rand_indices": idx in rand_indices,
                "rand_index_vectors_match_snapshot": bool(vectors_match),
                "mutant_matches_formula": bool(
                    np.allclose(
                        expected_mutant,
                        np.asarray(local_trace["mutant_vector"], dtype=np.float64),
                    )
                ),
            }
            self.debug_trace = local_trace


class RandomStrategyDifferentialEvolution(_RandomStrategySwitchingMixin, DifferentialEvolution):
    def _get_new_population(self) -> None:
        self._choose_and_apply_strategy()
        super()._get_new_population()


class RandomStrategyTracingDifferentialEvolution(_RandomStrategySwitchingMixin, TracingDifferentialEvolution):
    def _get_new_population(self) -> None:
        self._choose_and_apply_strategy()
        super()._get_new_population()


class RLDifferentialEvolution(TracingDifferentialEvolution):
    def __init__(
        self,
        *args: Any,
        rl_agent: QLearningAgent,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._rl_agent = rl_agent
        self.strategy_history_transitions: list[str] = []
        self.rl_history_transitions: list[dict[str, Any]] = []
        self.q_update_debug_trace: dict[str, Any] | None = None
        self._total_budget_calls = int(self._pop_size * self._iters)

    def _get_new_population(self) -> None:
        self._transition_index += 1

        current_state = budget_state_from_calls(self._calls, self._total_budget_calls)
        chosen_strategy = self._rl_agent.select_action(current_state)
        chosen_strategy = normalize_mutation_strategy(chosen_strategy)
        if chosen_strategy not in RANDOM_DE_STRATEGIES:
            raise ValueError(
                f"QLearningAgent returned '{chosen_strategy}', expected one of {RANDOM_DE_STRATEGIES}"
            )
        self._specified_mutation = chosen_strategy
        self.strategy_history_transitions.append(chosen_strategy)

        target_fitness_before = self._fitness_i.copy()
        population_before_mutation = self._population_g_i.copy()

        trial_population = np.empty_like(self._population_g_i, dtype=np.float64)
        local_trace: dict[str, Any] | None = None

        for i in range(self._pop_size):
            trial, trace = self._get_new_individ_g_with_trace(
                population_before_mutation[i], i, population_before_mutation
            )
            trial_population[i] = trial
            if trace is not None:
                local_trace = trace

        trial_population_ph = self._get_phenotype(trial_population)
        trial_fitness = self._get_fitness(trial_population_ph)
        reward, successful_replacements = strict_improvement_reward_from_internal_fitness(
            target_fitness_before, trial_fitness
        )
        mask = trial_fitness >= self._fitness_i

        self._population_g_i[mask] = trial_population[mask]
        self._population_ph_i[mask] = trial_population_ph[mask]
        self._fitness_i[mask] = trial_fitness[mask]

        is_terminal = self._transition_index == (self._iters - 1)
        next_state = budget_state_from_calls(self._calls, self._total_budget_calls)
        q_update_trace = self._rl_agent.update(
            state=current_state,
            action=chosen_strategy,
            reward=reward,
            next_state=None if is_terminal else next_state,
            done=is_terminal,
            debug=(
                self._debug_config.enabled
                and self._transition_index == self._debug_config.generation_to_trace
            ),
        )
        if q_update_trace is not None:
            self.q_update_debug_trace = q_update_trace

        self.rl_history_transitions.append(
            {
                "generation_transition": int(self._transition_index),
                "state": current_state,
                "action": chosen_strategy,
                "reward": float(reward),
                "successful_replacements": int(successful_replacements),
                "budget_calls_used": int(self._calls),
                "next_state": next_state,
                "done": bool(is_terminal),
                "q_table_after_update": self._rl_agent.get_q_table().tolist(),
            }
        )

        if local_trace is not None:
            idx = int(local_trace["individual_index"])
            target_internal = float(target_fitness_before[idx])
            trial_internal = float(trial_fitness[idx])
            selected_trial = bool(trial_internal >= target_internal)

            snapshot = np.asarray(local_trace["population_snapshot_before_mutation"], dtype=np.float64)
            best_vector = np.asarray(local_trace["best_vector_before_mutation"], dtype=np.float64)
            target_vector = np.asarray(local_trace["target_vector_before"], dtype=np.float64)
            rand_indices = [int(i) for i in local_trace["rand_indices"]]

            if local_trace["mutation_strategy"] == "rand_1":
                r1, r2, r3 = rand_indices
                x_r1 = snapshot[r1]
                x_r2 = snapshot[r2]
                x_r3 = snapshot[r3]
                expected_mutant = x_r3 + float(self._F) * (x_r1 - x_r2)
                expected_rand_keys = {"x_r1", "x_r2", "x_r3"}
                vectors_match = (
                    np.allclose(x_r1, np.asarray(local_trace["rand_vectors"]["x_r1"], dtype=np.float64))
                    and np.allclose(x_r2, np.asarray(local_trace["rand_vectors"]["x_r2"], dtype=np.float64))
                    and np.allclose(x_r3, np.asarray(local_trace["rand_vectors"]["x_r3"], dtype=np.float64))
                )
            elif local_trace["mutation_strategy"] == "best_1":
                r1, r2 = rand_indices
                x_r1 = snapshot[r1]
                x_r2 = snapshot[r2]
                expected_mutant = best_vector + float(self._F) * (x_r1 - x_r2)
                expected_rand_keys = {"x_r1", "x_r2"}
                vectors_match = (
                    np.allclose(x_r1, np.asarray(local_trace["rand_vectors"]["x_r1"], dtype=np.float64))
                    and np.allclose(x_r2, np.asarray(local_trace["rand_vectors"]["x_r2"], dtype=np.float64))
                )
            else:
                r1, r2 = rand_indices
                x_r1 = snapshot[r1]
                x_r2 = snapshot[r2]
                expected_mutant = (
                    target_vector
                    + float(self._F) * (best_vector - target_vector)
                    + float(self._F) * (x_r1 - x_r2)
                )
                expected_rand_keys = {"x_r1", "x_r2"}
                vectors_match = (
                    np.allclose(x_r1, np.asarray(local_trace["rand_vectors"]["x_r1"], dtype=np.float64))
                    and np.allclose(x_r2, np.asarray(local_trace["rand_vectors"]["x_r2"], dtype=np.float64))
                )

            local_trace["selection"] = {
                "target_value": float(self._sign * target_internal),
                "trial_value": float(self._sign * trial_internal),
                "selected": selected_trial,
                "reason": (
                    "trial_value <= target_value (minimization)"
                    if selected_trial
                    else "trial_value > target_value (minimization)"
                )
                if self._sign == -1
                else (
                    "trial_value >= target_value (maximization)"
                    if selected_trial
                    else "trial_value < target_value (maximization)"
                ),
            }
            local_trace["consistency_checks"] = {
                "rand_indices_unique": len(set(rand_indices)) == len(rand_indices),
                "rand_indices_in_range": all(0 <= i < len(snapshot) for i in rand_indices),
                "rand_vector_keys_match_strategy": set(local_trace["rand_vectors"].keys()) == expected_rand_keys,
                "target_index_present_in_rand_indices": idx in rand_indices,
                "rand_index_vectors_match_snapshot": bool(vectors_match),
                "mutant_matches_formula": bool(
                    np.allclose(
                        expected_mutant,
                        np.asarray(local_trace["mutant_vector"], dtype=np.float64),
                    )
                ),
            }
            self.debug_trace = local_trace


def _fitness_from_objective(
    objective: Callable[[NDArray[np.float64]], float]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    def _fitness(population: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray([objective(individ) for individ in population], dtype=np.float64)

    return _fitness


def _build_fitness_function(
    objective: Callable[[NDArray[np.float64]], float] | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    objective_mode: str,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    normalized_mode = normalize_objective_mode(objective_mode)
    if normalized_mode == "population":
        return objective  # type: ignore[return-value]
    return _fitness_from_objective(objective)  # type: ignore[arg-type]


def run_differential_evolution(
    objective: Callable[[NDArray[np.float64]], float] | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    config: DEConfig,
    debug_config: DEDebugConfig | None = None,
    strategy_selector: StrategySelector | None = None,
) -> DEResult:
    try:
        from importlib.metadata import version

        thefittest_version = version("thefittest")
    except Exception:
        thefittest_version = "unknown"

    debug_config = debug_config or DEDebugConfig(enabled=False)
    normalized_mutation = normalize_mutation_strategy(config.mutation)
    normalized_objective_mode = normalize_objective_mode(config.objective_mode)

    if normalized_mutation == "random_de":
        if debug_config.enabled:
            optimizer_cls: type[DifferentialEvolution] = RandomStrategyTracingDifferentialEvolution
            optimizer_kwargs: dict[str, Any] = {
                "debug_config": debug_config,
                "strategy_selector": strategy_selector,
                "strategy_random_seed": config.strategy_random_seed,
            }
        else:
            optimizer_cls = RandomStrategyDifferentialEvolution
            optimizer_kwargs = {
                "strategy_selector": strategy_selector,
                "strategy_random_seed": config.strategy_random_seed,
            }
    else:
        if debug_config.enabled:
            optimizer_cls = TracingDifferentialEvolution
            optimizer_kwargs = {"debug_config": debug_config}
        else:
            optimizer_cls = DifferentialEvolution
            optimizer_kwargs = {}

    fitness_function = _build_fitness_function(objective, normalized_objective_mode)

    best_value_history: list[float] = []

    def _on_generation(optimizer: DifferentialEvolution) -> None:
        best_value_history.append(float(optimizer._sign * optimizer.get_fittest()["fitness"]))

    optimizer = optimizer_cls(
        fitness_function=fitness_function,
        iters=config.iters,
        pop_size=config.pop_size,
        left_border=config.left_border,
        right_border=config.right_border,
        num_variables=config.num_variables,
        mutation=normalized_mutation,
        F=config.F,
        CR=config.CR,
        elitism=config.elitism,
        minimization=config.minimization,
        keep_history=True,
        random_state=config.seed,
        on_generation=_on_generation,
        **optimizer_kwargs,
    )
    optimizer.fit()

    if not best_value_history:
        best_value_history.append(float(optimizer._sign * optimizer.get_fittest()["fitness"]))
    else:
        initial_best = float(optimizer._sign * optimizer.get_stats()["max_fitness"][0])
        best_value_history = [initial_best] + best_value_history

    stats = optimizer.get_stats()
    population_history = [np.asarray(p, dtype=np.float64).copy() for p in stats.get("population_g", [])]
    best = optimizer.get_fittest()

    debug_trace: dict[str, Any] | None = None
    if isinstance(optimizer, TracingDifferentialEvolution):
        debug_trace = optimizer.debug_trace
        if debug_trace is not None and "fitness" in stats and "population_g" in stats:
            init_population = np.asarray(stats["population_g"][0], dtype=np.float64)
            init_internal_fitness = np.asarray(stats["fitness"][0], dtype=np.float64)
            init_values = (optimizer._sign * init_internal_fitness).astype(np.float64)
            init_best_idx = (
                int(np.argmin(init_values)) if config.minimization else int(np.argmax(init_values))
            )

            traced_transition = int(debug_trace["generation_transition"])
            generation_idx = traced_transition

            generation_summary: dict[str, Any] | None = None
            if generation_idx < len(stats["population_g"]):
                generation_population = np.asarray(stats["population_g"][generation_idx], dtype=np.float64)
                generation_internal_fitness = np.asarray(
                    stats["fitness"][generation_idx], dtype=np.float64
                )
                generation_values = optimizer._sign * generation_internal_fitness
                generation_best_idx = (
                    int(np.argmin(generation_values))
                    if config.minimization
                    else int(np.argmax(generation_values))
                )
                best_current = float(best_value_history[generation_idx])
                best_previous = float(best_value_history[generation_idx - 1]) if generation_idx > 0 else best_current
                generation_summary = {
                    "generation_index": generation_idx,
                    "best_vector": generation_population[generation_best_idx].tolist(),
                    "best_value": best_current,
                    "function_evaluations": int(config.pop_size * (generation_idx + 1)),
                    "delta_to_previous_generation": best_current - best_previous,
                }

            debug_trace = {
                "initialization": {
                    "population": init_population.tolist(),
                    "values": init_values.tolist(),
                    "best_index": init_best_idx,
                    "best_vector": init_population[init_best_idx].tolist(),
                    "best_value": float(init_values[init_best_idx]),
                },
                "mutation_crossover_selection": debug_trace,
                "generation_completion": generation_summary,
            }

    strategy_history_transitions: list[str]
    if hasattr(optimizer, "strategy_history_transitions"):
        strategy_history_transitions = list(getattr(optimizer, "strategy_history_transitions"))
    else:
        strategy_history_transitions = [normalized_mutation for _ in range(max(config.iters - 1, 0))]
    strategy_history = ["initialization"] + strategy_history_transitions

    return DEResult(
        config=config,
        best_solution=np.asarray(best["genotype"], dtype=np.float64),
        best_value=float(optimizer._sign * best["fitness"]),
        best_value_history=best_value_history,
        function_evaluations=int(optimizer._calls),
        population_history=population_history,
        strategy_history=strategy_history,
        debug_trace=debug_trace,
        thefittest_version=thefittest_version,
    )


def run_rl_differential_evolution(
    objective: Callable[[NDArray[np.float64]], float] | Callable[[NDArray[np.float64]], NDArray[np.float64]],
    config: DEConfig,
    rl_config: RLConfig = RLConfig(),
    debug_config: DEDebugConfig | None = None,
    agent: QLearningAgent | None = None,
) -> DEResult:
    try:
        from importlib.metadata import version

        thefittest_version = version("thefittest")
    except Exception:
        thefittest_version = "unknown"

    debug_config = debug_config or DEDebugConfig(enabled=False)
    normalized_objective_mode = normalize_objective_mode(config.objective_mode)

    if agent is None:
        q_agent = QLearningAgent(
            states=("early", "middle", "late"),
            actions=RANDOM_DE_STRATEGIES,
            alpha=rl_config.alpha,
            gamma=rl_config.gamma,
            epsilon=rl_config.epsilon,
            seed=rl_config.seed,
        )
    else:
        q_agent = agent
        q_agent.set_epsilon(rl_config.epsilon)
        if not rl_config.carry_learning:
            q_agent.reset(reset_rng=True)

    fitness_function = _build_fitness_function(objective, normalized_objective_mode)

    best_value_history: list[float] = []

    def _on_generation(optimizer: DifferentialEvolution) -> None:
        best_value_history.append(float(optimizer._sign * optimizer.get_fittest()["fitness"]))

    optimizer = RLDifferentialEvolution(
        fitness_function=fitness_function,
        iters=config.iters,
        pop_size=config.pop_size,
        left_border=config.left_border,
        right_border=config.right_border,
        num_variables=config.num_variables,
        mutation="rand_1",
        F=config.F,
        CR=config.CR,
        elitism=config.elitism,
        minimization=config.minimization,
        keep_history=True,
        random_state=config.seed,
        on_generation=_on_generation,
        debug_config=debug_config,
        rl_agent=q_agent,
    )
    optimizer.fit()

    if not best_value_history:
        best_value_history.append(float(optimizer._sign * optimizer.get_fittest()["fitness"]))
    else:
        initial_best = float(optimizer._sign * optimizer.get_stats()["max_fitness"][0])
        best_value_history = [initial_best] + best_value_history

    stats = optimizer.get_stats()
    population_history = [np.asarray(p, dtype=np.float64).copy() for p in stats.get("population_g", [])]
    best = optimizer.get_fittest()

    strategy_history = ["initialization"] + list(optimizer.strategy_history_transitions)
    rl_history = [dict(item) for item in optimizer.rl_history_transitions]
    for entry in rl_history:
        generation_idx = int(entry["generation_transition"])
        entry["best_value"] = float(best_value_history[generation_idx])

    debug_trace: dict[str, Any] | None = None
    if debug_config.enabled:
        debug_trace = {
            "de_trace": optimizer.debug_trace,
            "q_update_trace": optimizer.q_update_debug_trace,
        }

    return DEResult(
        config=config,
        best_solution=np.asarray(best["genotype"], dtype=np.float64),
        best_value=float(optimizer._sign * best["fitness"]),
        best_value_history=best_value_history,
        function_evaluations=int(optimizer._calls),
        population_history=population_history,
        strategy_history=strategy_history,
        rl_history=rl_history,
        final_q_table=q_agent.get_q_table().tolist(),
        debug_trace=debug_trace,
        thefittest_version=thefittest_version,
    )
