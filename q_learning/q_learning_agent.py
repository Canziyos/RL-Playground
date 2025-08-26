import random
from q_learning.model import GridWorld
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os

# hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
epsilon_decay = 0.999
episodes = 1000
max_steps = 30

# Create environment.
env = GridWorld()

# Initialize Q-table.
Q = {}
for row in range(env.grid_size[0]):
    for col in range(env.grid_size[1]):
        state = (row, col)
        Q[state] = {action: 0.0 for action in env.actions.keys()}

episode_rewards = []
track_state = (0, 0)
track_action = 'down_right'
tracked_q_values = []

# Training loop.
for episode in range(episodes):
    env.randomize_traps()
    state = env.initial_state
    total_reward = 0
    steps = 0

    while not env.is_terminal(state) and steps < max_steps:
        action = random.choice(list(env.actions)) if random.uniform(0, 1) < epsilon else max(Q[state], key=Q[state].get)
        new_state = env.get_next_state(state, action)
        reward = env.get_reward(new_state)
        total_reward += reward

        best_next_action = max(Q[new_state], key=Q[new_state].get)
        Q[state][action] += alpha * (reward + gamma * Q[new_state][best_next_action] - Q[state][action])

        if state == track_state and action == track_action:
            tracked_q_values.append(Q[state][action])

        state = new_state
        steps += 1

    episode_rewards.append(total_reward)
    epsilon = max(0.01, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} | Total reward: {total_reward:.2f}")

print("\nTraining complete!")

# Print Q-table
def print_q_table(Q):
    print("\nQ-table snapshot:")
    for state in sorted(Q):
        print(f"State {state}:")
        for action, value in Q[state].items():
            print(f"  {action:12} => Q = {value:.4f}")

print_q_table(Q)

# Track Q-value evolution.
print(f"\nQ-value evolution for {track_state} => '{track_action}':")
for i in range(0, len(tracked_q_values), max(1, len(tracked_q_values) // 10)):
    print(f"  Step {i+1}: Q = {tracked_q_values[i]:.4f}")
    env.print_grid(agent_pos=track_state)

# Final value
if tracked_q_values:
    print(f"\nFinal Q-value: {tracked_q_values[-1]:.4f}")
else:
    print(f"\n The tracked action '{track_action}' was never selected from {track_state}.")

# Greedy policy.
def run_greedy_policy(Q, env, max_steps=50):
    print("\nRunning greedy policy...")
    state = env.initial_state
    trajectory = [state]
    total_reward = 0

    for step in range(max_steps):
        if env.is_terminal(state):
            print(f"Reched terminal state: {state}")
            break

        action = max(Q[state], key=Q[state].get)
        next_state = env.get_next_state(state, action)
        reward = env.get_reward(next_state)
        total_reward += reward

        print(f"Step {step}: {state} => {action} => {next_state} | Reward: {reward}")
        trajectory.append(next_state)
        state = next_state

    else:
        print("Max steps reached without reaching terminal state.")

    print(f"Total greedy reward: {total_reward}")
    print("Final path taken:")
    print(" => ".join(str(s) for s in trajectory))

    return trajectory

# Visualization as GIF.
def save_gif_from_trajectory(env, trajectory, filename="agent_path.gif", cell_size=50, delay=0.5):
    frames = []
    width, height = env.grid_size[1] * cell_size, env.grid_size[0] * cell_size

    for step, agent_pos in enumerate(trajectory):
        fig, ax = plt.subplots(figsize=(env.grid_size[1], env.grid_size[0]))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Step {step}")

        for r in range(env.grid_size[0]):
            for c in range(env.grid_size[1]):
                x, y = c * cell_size, height - (r + 1) * cell_size
                rect = patches.Rectangle((x, y), cell_size, cell_size, linewidth=1, edgecolor='gray', facecolor='white')
                ax.add_patch(rect)

                state = (r, c)
                if state == env.terminal_state:
                    ax.add_patch(patches.Rectangle((x, y), cell_size, cell_size, facecolor='green'))
                elif state in env.walls:
                    ax.add_patch(patches.Rectangle((x, y), cell_size, cell_size, facecolor='black'))
                elif state in env.traps:
                    ax.add_patch(patches.Rectangle((x, y), cell_size, cell_size, facecolor='red'))
                elif state in env.teleporters:
                    ax.add_patch(patches.Rectangle((x, y), cell_size, cell_size, facecolor='blue'))

        # Draw agent.
        ax.add_patch(patches.Circle(
            (agent_pos[1] * cell_size + cell_size / 2, height - (agent_pos[0] + 0.5) * cell_size),
            cell_size / 4,
            facecolor='orange'
        ))

        fname = f"frame_{step:03d}.png"
        fig.savefig(fname)
        frames.append(fname)
        plt.close(fig)

    # Save gif.
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(filename, images, duration=delay)

    # Clean up.
    for f in frames:
        os.remove(f)

    print(f"GIF saved as: {filename}")
    return filename

# Run policy and save GIF.
trajectory = run_greedy_policy(Q, env)
save_gif_from_trajectory(env, trajectory, filename="agent_path.gif")
