import math
import os

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

from discretization import discretize_disturbed_state
from dynamics import THETA_LIMIT


def rollout_disturbed(agent, env, max_steps):
    state_cont = env.reset()
    state_disc = discretize_disturbed_state(state_cont)
    times = [0.0]
    positions = [state_cont[0]]
    angles = [state_cont[2]]
    done = False

    for _ in range(max_steps):
        action = agent.select_action(state_disc)
        env_action = -1 if action == 0 else 1
        
        state_cont, _, done = env.step(env_action)
        state_disc = discretize_disturbed_state(state_cont)
        times.append(times[-1] + env.dt)
        positions.append(state_cont[0])
        angles.append(state_cont[2])
        if done:
            break

    return times, positions, angles, done


def add_minutes_axis(ax):
    secax = ax.secondary_xaxis("top", functions=(lambda seconds: seconds / 60.0, lambda minutes: minutes * 60.0))
    secax.set_xlabel("Time (min)")
    secax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    return secax


def plot_rollout(times, positions, angles, done, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fail_t = times[-1] if done else None
    theta_limit_deg = math.degrees(THETA_LIMIT)
    theta_deg = [math.degrees(theta) for theta in angles]
    x_limit = 2.4

    fig1, ax1 = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax1.plot(times, positions, linewidth=1.6, label="x(t)")
    ax1.axhspan(-x_limit, x_limit, color="tab:blue", alpha=0.06, label="safe (|x| <= 2.4 m)")
    for sign in (1, -1):
        ax1.axhline(sign * x_limit, ls="--", lw=1.0, c="tab:blue")
    if fail_t:
        ax1.axvline(fail_t, c="tab:red", ls="--", lw=1.0, label="failure")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Cart position x (m)")
    ax1.set_title("Cart position under learned policy")
    add_minutes_axis(ax1)
    ax1.legend(loc="upper right")
    fig1.savefig(os.path.join(out_dir, "cart_position.png"))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax2.plot(times, theta_deg, lw=1.6, label="theta(t)")
    ax2.axhspan(-theta_limit_deg, theta_limit_deg, color="tab:orange", alpha=0.06, label=f"safe (|theta| <= {int(theta_limit_deg)} deg)")
    for sign in (1, -1):
        ax2.axhline(sign * theta_limit_deg, ls="--", lw=1.0, c="tab:orange")
    if fail_t:
        ax2.axvline(fail_t, c="tab:red", ls="--", lw=1.0, label="failure")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Pole angle theta (deg)")
    ax2.set_title("Pole angle under learned policy")
    add_minutes_axis(ax2)
    ax2.legend(loc="upper right")
    fig2.savefig(os.path.join(out_dir, "pole_angle.png"))
    plt.close(fig2)
