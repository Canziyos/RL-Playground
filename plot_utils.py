import matplotlib.pyplot as plt

def plot_rewards(rewards, title="Episode Rewards", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"📈 Plot saved to '{save_path}'")
    else:
        plt.show()
