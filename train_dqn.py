import gymnasium as gym
import random
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from dqn_model import DQNAgent
from replay_buffer import ReplayBuffer

def train_dqn():
    # Hyperparams.
    EPISODES = 3000
    GAMMA = 0.99
    LR = 1e-3
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQNAgent(input_dim, 256, output_dim).to(device)
    target_net = DQNAgent(input_dim, 256, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    loss_fn = torch.nn.SmoothL1Loss()  # Huber loss.
    epsilon = EPSILON_START

    episode_rewards = []
    episode_losses = []
    episode_steps = []

    for episode in range(EPISODES):
        # In gym≥0.26 return (obs, info) from reset.
        state, _ = env.reset()

        # Exampel: "custom state" (carefully!)
        custom_state = np.array([0.0, 0.0, np.deg2rad(50), 0.5])
        if env.observation_space.contains(custom_state):
            env.unwrapped.state = custom_state
            state = custom_state.copy()

        total_reward = 0.0
        losses = []

        for t in range(500):
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(s_tensor)
                    action = q_values.argmax(dim=1).item()

            # In gym≥0.26 return (obs, reward, done, truncated, info) from step.
            next_state, reward, done, truncated, _ = env.step(action)
            # combine 'done' & 'truncated' => terminate.
            is_done = done or truncated

            replay_buffer.store((state, action, reward, next_state, is_done))

            state = next_state
            total_reward += reward

            # when buffer is full => start training.
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards_, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_ = torch.FloatTensor(rewards_).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Q(s,a)
                q_values = policy_net(states).gather(1, actions)

                # Q-target (bootstrap från target_net).
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
                    targets = rewards_ + (1 - dones) * GAMMA * max_next_q

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                # grad-clipping (optional).
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                losses.append(loss.item())

            if is_done:
                break

        # Uppdate epsilon (explore)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Sync target_net ocationally.
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Loggning.
        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(losses) if losses else 0)
        episode_steps.append(t)

        if (episode + 1) % 50 == 0:
            print(f"Ep {episode+1} | Reward: {total_reward:.1f} | Eps: {epsilon:.3f} "
                  f"| Avg Loss: {episode_losses[-1]:.4f}")

    # Save.
    torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
    print("Modell sparad i 'dqn_cartpole.pth'")

    # Plot.
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("rewards_plot.png")
    plt.show()

if __name__ == "__main__":
    train_dqn()
    print("Training Done!")