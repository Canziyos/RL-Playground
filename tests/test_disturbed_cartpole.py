import json
from pathlib import Path

import pytest

from src.envs.disturbed_cartpole import DisturbedCartPoleSim


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "disturbed.json"


def load_env_params():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)["env"]


def test_disturbed_env_works_without_disturbances():
    env_params = load_env_params()

    env = DisturbedCartPoleSim(**env_params, disturbances={})
    state = env.reset(drunk=False)

    next_state, reward, done = env.step(1)

    assert len(state) == 4
    assert len(next_state) == 4
    assert reward in (0, 1)
    assert isinstance(done, bool)


def test_disturbed_env_rejects_invalid_action():
    env_params = load_env_params()

    env = DisturbedCartPoleSim(**env_params, disturbances={})
    env.reset()

    with pytest.raises(ValueError):
        env.step(2)