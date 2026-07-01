from collections import deque
import random

from agent import QAgent
from discretization import discretize_base_state, discretize_disturbed_state
from envs.base_cartpole import BaseCartPoleSim
from envs.disturbed_cartpole import DisturbedCartPoleSim


def require_env_params(env_params):
    if env_params is None:
        raise ValueError(
            "env_params must be provided. Load the config in the script and pass config['env']."
        )
    return env_params


def train_base_agent(
    episodes,
    max_steps,
    alpha,
    gamma,
    epsilon,
    env_params,
    log_every=0,
    seed=None,
):
    env_params = require_env_params(env_params)

    env = BaseCartPoleSim(**env_params)
    agent = QAgent(n_actions=2, alpha=alpha, gamma=gamma, epsilon=epsilon)

    if seed is not None:
        agent.rng.seed(int(seed))

    episode_lengths = []

    for episode in range(episodes):
        state = discretize_base_state(env.reset())
        steps = 0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_cont, reward, done = env.step(action)
            next_state = discretize_base_state(next_cont)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1

            if done:
                break

        episode_lengths.append(steps)

        if log_every and (episode + 1) % log_every == 0:
            last = episode_lengths[-log_every:]
            avg = sum(last) / len(last)
            print(f"Episode {episode + 1:4d} | steps={steps:5d} | avg(last {log_every})={avg:7.1f}")

    return agent, episode_lengths


def train_disturbed_agent(
    episodes,
    max_steps,
    alpha,
    gamma,
    epsilon,
    env_params,
    drunk=False,
    log_every=0,
    reverse_update=False,
    disturbances=None,
    seed=None,
):
    env_params = require_env_params(env_params)

    rng = random.Random(int(seed)) if seed is not None else None

    env = DisturbedCartPoleSim(
        **env_params,
        disturbances=disturbances,
        rng=rng,
    )

    agent = QAgent(n_actions=2, alpha=alpha, gamma=gamma, epsilon=epsilon)

    if seed is not None:
        agent.rng.seed(int(seed))

    last = deque(maxlen=log_every) if log_every else None

    for episode in range(episodes):
        state = discretize_disturbed_state(env.reset(drunk=drunk))
        steps = 0
        trajectory = []

        while steps < max_steps:
            action = agent.select_action(state)
            env_action = -1 if action == 0 else 1

            next_cont, reward, done = env.step(env_action)
            next_state = discretize_disturbed_state(next_cont)

            if reverse_update:
                trajectory.append((state, action, reward, next_state, done))
            else:
                agent.update(state, action, reward, next_state, done)

            state = next_state
            steps += 1

            if done:
                break

        if reverse_update and trajectory:
            for state, action, reward, next_state, done in reversed(trajectory):
                agent.update(state, action, reward, next_state, done)

        if last is not None:
            last.append(steps)

            if (episode + 1) % log_every == 0:
                avg = sum(last) / len(last)
                print(f"Episode {episode + 1:4d} | steps={steps:4d} | avg(last {log_every})={avg:6.1f}")

    return agent
