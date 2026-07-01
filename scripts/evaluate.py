import argparse
import json
import os
import pickle
import random

from src.agent import QAgent
from src.envs.base_cartpole import BaseCartPoleSim
from src.envs.disturbed_cartpole import DisturbedCartPoleSim
from src.evaluation import evaluate_base_policy, evaluate_disturbed_policy


CONFIG_PATHS = {
    "base": "configs/base.json",
    "disturbed": "configs/disturbed.json",
}

Q_TABLE_PATHS = {
    "base": "outputs/base/q_table.pkl",
    "disturbed": "outputs/disturbed/q_table.pkl",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_q_table(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def make_agent(config, q_table):
    eval_cfg = config["evaluation"]

    agent = QAgent(
        n_actions=2,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=eval_cfg["epsilon"],
    )
    agent.Q = q_table
    return agent


def make_base_env(config):
    env_cfg = config["env"]

    return BaseCartPoleSim(
        cart_mass=env_cfg["cart_mass"],
        pole_mass=env_cfg["pole_mass"],
        length=env_cfg["length"],
        gravity=env_cfg["gravity"],
        force=env_cfg["force"],
        dt=env_cfg["dt"],
    )


def make_disturbed_env(config):
    return DisturbedCartPoleSim(
        **config["env"],
        disturbances=config["eval_disturbances"],
        rng=random.Random(config["eval_seed"]),
    )


def evaluate_base(config, agent):
    env = make_base_env(config)
    eval_cfg = config["evaluation"]

    steps, reason = evaluate_base_policy(
        agent,
        env,
        cap_steps=eval_cfg["cap_steps"],
        logs=eval_cfg["logs"],
    )

    print(f"Base survival: {steps} steps ({steps * env.dt:.1f} s). Stopped: {reason}.")


def evaluate_disturbed(config, agent):
    env = make_disturbed_env(config)
    eval_cfg = config["evaluation"]

    steps, reason = evaluate_disturbed_policy(
        agent,
        env,
        cap_steps=eval_cfg["cap_steps"],
        drunk=bool(config["eval_drunk_initial_state"]),
        logs=eval_cfg["logs"],
    )

    print(f"Disturbed survival: {steps} steps ({steps * env.dt:.1f} s). Stopped: {reason}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CartPole Q-learning policy.")
    parser.add_argument(
        "mode",
        choices=("base", "disturbed"),
        help="Which trained policy/environment to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode

    config_path = CONFIG_PATHS[mode]
    q_table_path = Q_TABLE_PATHS[mode]

    if not os.path.exists(config_path):
        raise SystemExit(f"Missing config file: {config_path}")

    if not os.path.exists(q_table_path):
        raise SystemExit(f"Missing {q_table_path}. Run the corresponding training script first.")

    config = load_json(config_path)
    q_table = load_q_table(q_table_path)
    agent = make_agent(config, q_table)

    if mode == "base":
        evaluate_base(config, agent)
    elif mode == "disturbed":
        evaluate_disturbed(config, agent)
    else:
        raise SystemExit(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()