import json
import os
import random
from itertools import product

from src.envs.disturbed_cartpole import DisturbedCartPoleSim
from src.evaluation import evaluate_disturbed_policy
from src.training import train_disturbed_agent


CONFIG_PATH = "configs/disturbed_search.json"
TARGET_CONFIG_PATH = "configs/disturbed.json"
OUTPUT_DIR = "outputs/disturbed"


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


def make_disturbed_env(config):
    eval_cfg = config["evaluation"]

    return DisturbedCartPoleSim(
        **config["env"],
        disturbances=config["eval_disturbances"],
        rng=random.Random(eval_cfg["eval_seed"]),
    )


def sort_results(results, rank_by):
    return sorted(
        results,
        key=lambda row: tuple(row[key] for key in rank_by),
        reverse=True,
    )


def build_best_config(best_result, training_cfg, eval_cfg, env_cfg, config, current_config):
    best_train_disturbances = config["train_disturbances"][
        best_result["train_disturbance_profile"]
    ]

    best_config = {
        "alpha": best_result["alpha"],
        "gamma": best_result["gamma"],
        "epsilon": best_result["epsilon"],
        "episodes": best_result["episodes"],
        "max_steps": training_cfg["max_steps"],
        "train_seed": training_cfg["train_seed"],
        "eval_seed": eval_cfg["eval_seed"],
        "log_every": training_cfg["log_every"],
        "drunk_initial_state": best_result["drunk_initial_state"],
        "eval_drunk_initial_state": eval_cfg["eval_drunk_initial_state"],
        "reverse_update": best_result["reverse_update"],
        "env": env_cfg,
        "train_disturbances": best_train_disturbances,
        "eval_disturbances": config["eval_disturbances"],
        "evaluation": {
            "cap_steps": eval_cfg["cap_steps"],
            "epsilon": eval_cfg["epsilon"],
            "logs": eval_cfg["logs"],
        },
        "score": {
            "survival": best_result["survival"],
            "reason": best_result["reason"],
            "train_disturbance_profile": best_result["train_disturbance_profile"],
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
        search_cfg["drunk_initial_state"],
        search_cfg["reverse_update"],
        search_cfg["train_disturbance_profiles"],
    )

    for idx, (
        alpha,
        gamma,
        epsilon,
        episodes,
        drunk_initial_state,
        reverse_update,
        train_profile_name,
    ) in enumerate(candidates, start=1):

        train_disturbances = config["train_disturbances"][train_profile_name]

        print(
            f"\nConfig {idx:3d}: "
            f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}, episodes={episodes}, "
            f"drunk_initial_state={drunk_initial_state}, "
            f"reverse_update={reverse_update}, "
            f"train_profile={train_profile_name}."
        )

        agent = train_disturbed_agent(
            episodes=int(episodes),
            max_steps=int(training_cfg["max_steps"]),
            alpha=float(alpha),
            gamma=float(gamma),
            epsilon=float(epsilon),
            drunk=bool(drunk_initial_state),
            log_every=int(training_cfg["log_every"]),
            reverse_update=bool(reverse_update),
            disturbances=train_disturbances,
            env_params=env_cfg,
            seed=training_cfg["train_seed"],
        )

        agent.epsilon = float(eval_cfg["epsilon"])

        env = make_disturbed_env(config)

        steps, reason = evaluate_disturbed_policy(
            agent,
            env,
            cap_steps=int(eval_cfg["cap_steps"]),
            drunk=bool(eval_cfg["eval_drunk_initial_state"]),
            logs=int(eval_cfg["logs"]),
        )

        result = {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "episodes": episodes,
            "drunk_initial_state": drunk_initial_state,
            "reverse_update": reverse_update,
            "train_disturbance_profile": train_profile_name,
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
        config=config,
        current_config=current_config,
    )

    output = {
        "source": CONFIG_PATH,
        "target_config": TARGET_CONFIG_PATH,
        "rank_by": eval_cfg["rank_by"],
        "eval_disturbances": config["eval_disturbances"],
        "best": best_config,
        "results": ranked_results,
    }

    save_json(os.path.join(OUTPUT_DIR, "grid_search_results.json"), output)
    save_json(TARGET_CONFIG_PATH, best_config)

    print(f"\nSaved full disturbed grid-search results to {OUTPUT_DIR}.")
    print(f"Updated best disturbed config: {TARGET_CONFIG_PATH}.")


if __name__ == "__main__":
    main()