import hashlib
import json
import os
import pickle
from datetime import datetime

from src.training import train_base_agent


CONFIG_PATH = "configs/base.json"
OUTPUT_DIR = "outputs/base"

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    agent, lengths = train_base_agent(
        episodes=int(config["episodes"]),
        max_steps=int(config["max_steps"]),
        alpha=float(config["alpha"]),
        gamma=float(config["gamma"]),
        epsilon=float(config["epsilon"]),
        env_params=config["env"],
        log_every=int(config["log_every"]),
        seed=config["seed"],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Latest/default files for simple evaluation.
    latest_q_table_path = os.path.join(OUTPUT_DIR, "q_table.pkl")
    latest_lengths_path = os.path.join(OUTPUT_DIR, "episode_lengths.json")

    with open(latest_q_table_path, "wb") as file:
        pickle.dump(agent.Q, file)

    save_json(latest_lengths_path, lengths)

    # Archived run files for experiment tracking.
    run_dir = os.path.join(OUTPUT_DIR)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "q_table.pkl"), "wb") as file:
        pickle.dump(agent.Q, file)

    save_json(os.path.join(run_dir, "episode_lengths.json"), lengths)
    save_json(os.path.join(run_dir, "config.json"), config)

    print(f"Saved latest base Q-table to {latest_q_table_path}.")
    print(f"Saved archived run to {run_dir}.")


if __name__ == "__main__":
    main()