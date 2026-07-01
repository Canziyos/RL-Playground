import json
import pickle
from datetime import timedelta

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.agent import QAgent
from src.discretization import discretize_base_state
from src.envs.base_cartpole import BaseCartPoleSim


CONFIG_PATH = "configs/base.json"
Q_TABLE_PATH = "outputs/base/q_table.pkl"

N_ACTIONS = 2

X_LIMIT = 2.4
Y_LIMIT = 1.2

GROUND_Y = -0.2
CART_WIDTH = 0.3
CART_HEIGHT = 0.2
CART_BASE_OFFSET = 0.1

WHEEL_RADIUS = 0.05
WHEEL_X_OFFSET_FACTOR = 4

POLE_LINE_WIDTH = 3
GROUND_LINE_WIDTH = 2
POLE_RENDER_SCALE = 2.0

TIME_TEXT_X = 2.3
TIME_TEXT_Y = 1.1
TIME_TEXT_SIZE = 10


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_q_table(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def make_agent(config, q_table):
    eval_cfg = config["evaluation"]

    agent = QAgent(
        n_actions=N_ACTIONS,
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=eval_cfg["epsilon"],
    )
    agent.Q = q_table
    return agent


def make_base_env(env_cfg):
    return BaseCartPoleSim(
        cart_mass=env_cfg["cart_mass"],
        pole_mass=env_cfg["pole_mass"],
        length=env_cfg["length"],
        gravity=env_cfg["gravity"],
        force=env_cfg["force"],
        dt=env_cfg["dt"],
    )


def main():
    config = load_json(CONFIG_PATH)
    q_table = load_q_table(Q_TABLE_PATH)

    agent = make_agent(config, q_table)
    env = make_base_env(config["env"])

    vis_cfg = config["visualization"]

    state_cont = env.reset()
    state_disc = discretize_base_state(state_cont)

    fig, ax = plt.subplots()
    ax.set_xlim(-X_LIMIT, X_LIMIT)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)
    ax.set_aspect("equal")
    ax.set_title("Base CartPole")

    ax.plot(
        [-X_LIMIT, X_LIMIT],
        [GROUND_Y, GROUND_Y],
        "k-",
        lw=GROUND_LINE_WIDTH,
    )

    cart_patch = plt.Rectangle(
        (state_cont[0] - CART_WIDTH / 2, GROUND_Y + CART_BASE_OFFSET),
        CART_WIDTH,
        CART_HEIGHT,
        color="black",
    )

    (pole_line,) = ax.plot([], [], lw=POLE_LINE_WIDTH)

    left_wheel = plt.Circle(
        (
            state_cont[0] - CART_WIDTH / WHEEL_X_OFFSET_FACTOR,
            GROUND_Y + WHEEL_RADIUS,
        ),
        WHEEL_RADIUS,
    )

    right_wheel = plt.Circle(
        (
            state_cont[0] + CART_WIDTH / WHEEL_X_OFFSET_FACTOR,
            GROUND_Y + WHEEL_RADIUS,
        ),
        WHEEL_RADIUS,
    )

    time_text = ax.text(
        TIME_TEXT_X,
        TIME_TEXT_Y,
        "t = 00:00:00",
        ha="right",
        va="top",
        fontsize=TIME_TEXT_SIZE,
    )

    ax.add_patch(cart_patch)
    ax.add_patch(left_wheel)
    ax.add_patch(right_wheel)

    elapsed = 0.0

    def draw(state):
        x, _, theta, _ = state

        cart_patch.set_xy(
            (
                x - CART_WIDTH / 2,
                GROUND_Y + CART_BASE_OFFSET,
            )
        )

        left_wheel.center = (
            x - CART_WIDTH / WHEEL_X_OFFSET_FACTOR,
            GROUND_Y + WHEEL_RADIUS,
        )

        right_wheel.center = (
            x + CART_WIDTH / WHEEL_X_OFFSET_FACTOR,
            GROUND_Y + WHEEL_RADIUS,
        )

        pole_base_y = GROUND_Y + CART_BASE_OFFSET + CART_HEIGHT / 2

        x_top = x + env.l * POLE_RENDER_SCALE * np.sin(theta)
        y_top = pole_base_y + env.l * POLE_RENDER_SCALE * np.cos(theta)

        pole_line.set_data([x, x_top], [pole_base_y, y_top])

    def init():
        draw(state_cont)
        return cart_patch, pole_line, left_wheel, right_wheel, time_text

    def update(frame):
        nonlocal state_cont, state_disc, elapsed

        action = agent.select_action(state_disc)
        state_cont, _, done = env.step(action)
        state_disc = discretize_base_state(state_cont)

        draw(state_cont)

        elapsed += env.dt
        time_text.set_text(f"t = {timedelta(seconds=int(elapsed))}")

        if done:
            time_text.set_color("red")
            anim.event_source.stop()
            print(f"Stopped at frame {frame}.")

        return cart_patch, pole_line, left_wheel, right_wheel, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=int(vis_cfg["max_steps"]),
        interval=int(env.dt * 1000),
        blit=False,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    main()