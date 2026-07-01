import hashlib
import json
import os
import pickle
from src.training import train_disturbed_agent

CONFIG_PATH = "configs/disturbed.json"
OUTPUT_DIR = "outputs/disturbed"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_q_table(path, q_table):
    with open(path, "wb") as file:
        pickle.dump(q_table, file)


def main():
    config = load_json(CONFIG_PATH)

    agent = train_disturbed_agent(
        episodes=int(config["episodes"]),
        max_steps=int(config["max_steps"]),
        alpha=float(config["alpha"]),
        gamma=float(config["gamma"]),
        epsilon=float(config["epsilon"]),
        drunk=bool(config["drunk_initial_state"]),
        log_every=int(config["log_every"]),
        reverse_update=bool(config["reverse_update"]),
        disturbances=config["train_disturbances"],
        env_params=config["env"],
        seed=config["train_seed"],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Latest/default file used by evaluation and visualization.
    latest_q_table_path = os.path.join(OUTPUT_DIR, "q_table.pkl")
    save_q_table(latest_q_table_path, agent.Q)

    # Archived run for experiment tracking.
    run_dir = os.path.join(OUTPUT_DIR)
    os.makedirs(run_dir, exist_ok=True)

    save_q_table(os.path.join(run_dir, "q_table.pkl"), agent.Q)
    save_json(os.path.join(run_dir, "config.json"), config)

    print(f"Saved latest disturbed Q-table to {latest_q_table_path}.")
    print(f"Saved archived disturbed run to {run_dir}.")


if __name__ == "__main__":
    main()