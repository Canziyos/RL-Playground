import json
import os
from itertools import product
from statistics import mean

from src.envs.base_cartpole import BaseCartPoleSim
from src.evaluation import evaluate_base_policy
from src.training import train_base_agent


CONFIG_PATH = "configs/base_search.json"
TARGET_CONFIG_PATH = "configs/base.json"
OUTPUT_DIR = "outputs/base"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json_if_exists(path):
    if not os.path.exists(path):
        return {}

    return load_json(path)


def make_base_env(env_cfg):
    return BaseCartPoleSim(**env_cfg)


def sort_results(results, rank_by):
    return sorted(
        results,
        key=lambda row: tuple(row[key] for key in rank_by),
        reverse=True,
    )


def build_best_config(best_result, training_cfg, eval_cfg, env_cfg, current_config):
    best_config = {
        "alpha": best_result["alpha"],
        "gamma": best_result["gamma"],
        "epsilon": best_result["epsilon"],
        "episodes": best_result["episodes"],
        "max_steps": training_cfg["max_steps"],
        "seed": training_cfg["seed"],
        "log_every": training_cfg["log_every"],
        "env": env_cfg,
        "evaluation": {
            "cap_steps": eval_cfg["cap_steps"],
            "epsilon": eval_cfg["epsilon"],
            "logs": eval_cfg["logs"],
        },
        "score": {
            "best": best_result["best"],
            "mean": best_result["mean"],
            "survival": best_result["survival"],
            "reason": best_result["reason"],
        },
    }

    if "visualization" in current_config:
        best_config["visualization"] = current_config["visualization"]

    return best_config


def main():
    config = load_json(CONFIG_PATH)
    current_config = load_json_if_exists(TARGET_CONFIG_PATH)

    search_cfg = config["search"]
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]
    env_cfg = config["env"]

    results = []

    candidates = product(
        search_cfg["alphas"],
        search_cfg["gammas"],
        search_cfg["epsilons"],
        search_cfg["episodes"],
    )

    for idx, (alpha, gamma, epsilon, episodes) in enumerate(candidates, start=1):
        print(
            f"\nConfig {idx:3d}: "
            f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={episodes}."
        )

        agent, lengths = train_base_agent(
            episodes=int(episodes),
            max_steps=int(training_cfg["max_steps"]),
            alpha=float(alpha),
            gamma=float(gamma),
            epsilon=float(epsilon),
            env_params=env_cfg,
            log_every=int(training_cfg["log_every"]),
            seed=training_cfg["seed"],
        )

        agent.epsilon = float(eval_cfg["epsilon"])

        steps, reason = evaluate_base_policy(
            agent,
            make_base_env(env_cfg),
            cap_steps=int(eval_cfg["cap_steps"]),
            logs=int(eval_cfg["logs"]),
        )

        result = {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "episodes": episodes,
            "best": max(lengths) if lengths else 0,
            "mean": mean(lengths) if lengths else 0.0,
            "survival": steps,
            "reason": reason,
        }

        results.append(result)

        print(f"Result: survival={steps} ({reason}).")

    ranked_results = sort_results(results, eval_cfg["rank_by"])
    best_result = ranked_results[0]

    best_config = build_best_config(
        best_result=best_result,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        env_cfg=env_cfg,
        current_config=current_config,
    )

    output = {
        "source": CONFIG_PATH,
        "target_config": TARGET_CONFIG_PATH,
        "rank_by": eval_cfg["rank_by"],
        "best": best_config,
        "results": ranked_results,
    }

    save_json(os.path.join(OUTPUT_DIR, "grid_search_results.json"), output)
    save_json(TARGET_CONFIG_PATH, best_config)

    print(f"\nSaved full grid-search results to {OUTPUT_DIR}.")
    print(f"Updated best base config: {TARGET_CONFIG_PATH}.")


if __name__ == "__main__":
    main()