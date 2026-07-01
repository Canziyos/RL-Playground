import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from dqn_model import DQNAgent
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

RENDER = True  # Set to True to render in human mode.
MODEL_PATH = "dqn_cartpole.pth"
#env = gym.make("CartPole-v1", render_mode="rgb_array")

env = CartPoleEnv()
env.render_mode = "rgb_array" if not RENDER else "human"
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_net = DQNAgent(input_dim, 128, output_dim)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()

frames = []
custom_state = np.array([0.0, 0.0, np.deg2rad(30), -0.5])
if env.observation_space.contains(custom_state):
    env.unwrapped.state = custom_state
    state = custom_state.copy()
else:
    state = env.reset()[0]

total_reward = 0

t = 0
while True:
    frame = env.render()
    frames.append(frame)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = torch.argmax(policy_net(state_tensor)).item()

    next_state, reward, done, _, _ = env.step(action)
    total_reward += reward
    state = next_state
    t += 1

    if done:
        break

env.close()
gif_filename = "cartpole_run.gif"
imageio.mimsave(gif_filename, frames, duration=1/30)
print(f"Saved agent trajectory to: {gif_filename}")
print(f"Total reward: {total_reward}")
print(f"Total steps: {t + 1}")
