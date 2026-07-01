import torch
import gymnasium as gym
import numpy as np
import imageio
import csv
import os
import matplotlib.pyplot as plt

# Same arch (256)
from dqn_model import DQNAgent

MODEL_PATH = "dqn_cartpole.pth"
EPISODES = 5
RENDER = True  # True = GUI, False = render_mode="rgb_array"

def main():
    # 1) create env without limits.
    env = gym.make(
        "CartPole-v1",
        render_mode="human" if RENDER else "rgb_array"
    )
    # Gym-limiter set high or infint.
    env._max_episode_steps = 9999999  

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    # 2) The network we tarined (hidden_dim=256).
    policy_net = DQNAgent(input_dim, 256, output_dim)
    policy_net.load_state_dict(torch.load(MODEL_PATH))
    policy_net.eval()
    
    os.makedirs("runs", exist_ok=True)
    step_log = []
    all_rewards = []

    for ep in range(1, EPISODES + 1):
        # Gym≥0.26: reset() return (obs, info)
        state, _ = env.reset()

        # -- Slumpa ett “extra svårt” startläge --        
        #   Note: CartPole "done" om
        #   x are outside [-4.8, 4.8] or
        #   vinkel outof range ~±0.209 rad (~12 grades)
        #   Så välj något i rimligt intervall!
        x = np.random.uniform(-2.4, 2.4)
        x_dot = np.random.uniform(-2.0, 2.0)     # Faster start
        theta = np.random.uniform(-0.35, 0.35)   # upp to ~20 grades
        theta_dot = np.random.uniform(-2.0, 2.0) # snabbare rotation

        env.unwrapped.state = np.array([x, x_dot, theta, theta_dot])
        state = env.unwrapped.state.copy()

        total_reward = 0.0
        frames = []
        t = 0

        # 3) Run untill done/truncated.
        while True:
            if not RENDER:
                frames.append(env.render())
            
            #Pick a procedure from the träined net.
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            
            # next_state, reward, done, truncated, _ = env.step(action)
            # total_reward += reward
            
            # # (Valfritt) Lägg till en slumpmässig "vind" i vagnen:
            # #  - Här "hackar" vi in lite disturbance på x_dot
            # #    efter steget, innan vi uppdaterar state:
            # disturbance = np.random.uniform(-0.2, 0.2)
            # s = env.unwrapped.state
            # s[1] += disturbance   # change the velocity.
            # env.unwrapped.state = s
            # next_state = env.unwrapped.state.copy()
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            disturbance = np.random.uniform(-0.2, 0.2)
            arr_s = np.array(env.unwrapped.state, dtype=np.float32)
            arr_s[1] += disturbance  # manipulate car velocity.
            env.unwrapped.state = arr_s
            next_state = env.unwrapped.state.copy()
            step_log.append([
                ep, t, *state, action, reward,
                *next_state, (done or truncated)
            ])
            
            state = next_state
            t += 1
            if done or truncated:
                break
        
        print(f"Episode {ep} — Total reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

        if not RENDER:
            gif_path = f"runs/cartpole_ep{ep}.gif"
            imageio.mimsave(gif_path, frames, duration=0.03)
            print(f"GIF saved: {gif_path}")

    env.close()

    # Save all thing into CSV (for now).
    csv_path = "runs/cartpole_test_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode", "Step", "x", "x_dot", "theta", "theta_dot", 
            "Action", "Reward", 
            "Next_x", "Next_x_dot", "Next_theta", "Next_theta_dot", 
            "Done"
        ])
        writer.writerows(step_log)
    print(f"Log saved to: {csv_path}")

    mean_reward = np.mean(all_rewards)
    print(f"\nMean reward over {EPISODES} episodes: {mean_reward:.2f}")

    plt.plot(all_rewards)
    plt.title("Reward per Test Episode - Disturbed CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.savefig("runs/test_rewards_plot.png")
    plt.show()
    print("Plot saved to 'runs/test_rewards_plot.png'")

if __name__ == "__main__":
    main()
