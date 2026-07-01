import json
from pathlib import Path

from src.envs.base_cartpole import BaseCartPoleSim


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.json"


def load_env_params():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["env"]


def test_base_env_reset_returns_four_state_tuple():
    env_params = load_env_params()

    env = BaseCartPoleSim(**env_params)
    state = env.reset()

    assert isinstance(state, tuple)
    assert len(state) == 4


def test_base_env_step_returns_state_reward_done():
    env_params = load_env_params()

    env = BaseCartPoleSim(**env_params)
    env.reset()

    state, reward, done = env.step(1)

    assert len(state) == 4
    assert reward in (0, 1)
    assert isinstance(done, bool)