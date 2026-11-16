from __future__ import annotations

import argparse

import numpy as np

from rl_optimization.agents.q_learning import QLearningAgent


STATES = ("early", "middle", "late")
OPTIMAL_ACTION_BY_STATE = {
    "early": "rand_1",
    "middle": "best_1",
    "late": "current_to_best_1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q-learning demo on an artificial environment")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-start", type=float, default=0.8)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-steps", type=int, default=6)
    return parser.parse_args()


def reward_for(state: str, action: str) -> float:
    return 1.0 if OPTIMAL_ACTION_BY_STATE[state] == action else 0.0


def linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    frac = step / (total_steps - 1)
    return start + (end - start) * frac


def main() -> None:
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True)

    agent = QLearningAgent(
        states=STATES,
        actions=("rand_1", "best_1", "current_to_best_1"),
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        seed=args.seed,
    )

    print("=== Q-learning demo (artificial environment) ===")
    print("Initial Q-table:")
    print(agent.get_q_table())
    print()

    debug_printed = 0
    for episode in range(args.episodes):
        agent.set_epsilon(
            linear_decay(args.epsilon_start, args.epsilon_end, episode, args.episodes)
        )
        for idx, state in enumerate(STATES):
            action = agent.select_action(state)
            reward = reward_for(state, action)
            next_state = STATES[idx + 1] if idx < len(STATES) - 1 else None
            done = idx == len(STATES) - 1
            trace = agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                debug=debug_printed < args.debug_steps,
            )
            if trace is not None:
                print(
                    f"[debug] s={trace['state']}, a={trace['action']}, r={trace['reward']}, "
                    f"s'={trace['next_state']}, done={trace['done']}"
                )
                print(
                    f"        old_q={trace['old_q']:.4f}, max_next_q={trace['max_next_q']:.4f}, "
                    f"td_error={trace['td_error']:.4f}, new_q={trace['new_q']:.4f}"
                )
                print(f"        formula: {trace['formula']}")
                if trace["terminal_transition"]:
                    print("        terminal: future reward is not used.")
                debug_printed += 1

    print()
    print("Final Q-table:")
    print(agent.get_q_table())
    print()

    print("Preferred actions by state:")
    for state in STATES:
        print(f"  {state}: {agent.get_preferred_action(state)}")
    print(f"Total updates: {agent.update_count}")


if __name__ == "__main__":
    main()
