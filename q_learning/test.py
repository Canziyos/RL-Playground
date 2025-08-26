import random
from q_learning.model import GridWorld

episodes = 10  # number of times the agent will try.
max_steps = 30

for episode in range(episodes):
    env = GridWorld()
    state = env.initial_state
    total_reward = 0
    steps = 0

    print(f"\nEpisode {episode + 1} starts at {state}")

    while not env.is_terminal(state) and steps < max_steps:
        action = random.choice(list(env.actions.keys()))
        new_state = env.get_next_state(state, action)
        reward = env.get_reward(new_state)
        total_reward += reward

        print(f"  Step {steps}: {action} => {new_state} | Reward: {reward}")
        state = new_state
        steps += 1

    if env.is_terminal(state):
        print("  Reached terminal state!")
    else:
        print("  Max steps reached.")

    print(f"Total Reward: {total_reward}")
