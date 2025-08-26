import torch
import gym
import numpy as np
import imageio
from dqn_model import DQNAgent

MODEL_PATH = "dqn_cartpole.pth"
RENDER = True

from gym.envs.classic_control.cartpole import CartPoleEnv
env = CartPoleEnv()
env.render_mode = "rgb_array" if not RENDER else "human"

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_net = DQNAgent(input_dim, 128, output_dim)
policy_net.load_state_dict(torch.load(MODEL_PATH))
policy_net.eval()

initial_state = np.array([0.0, 0.0, np.deg2rad(10), 0.5])
env.unwrapped.state = initial_state.copy()
state = initial_state.copy()

frames = []
total_reward = 0
t = 0

while True:
    if not RENDER:
        frames.append(env.render())

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

if not RENDER:
    imageio.mimsave("cartpole_one_run.gif", frames, duration=1/30)
    print("GIF saved to cartpole_one_run.gif")

print(f"Total reward: {total_reward}")
print(f"Total steps: {t}")
