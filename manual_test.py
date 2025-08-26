import torch
import torch.nn as nn
from dqn_model import DQNAgent

# Initiate model and optimize.
policy_net = DQNAgent()
target_net = DQNAgent()
target_net.load_state_dict(policy_net.state_dict())  # Same weights in the beginning.
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Simulated experience.
state = torch.tensor([[0.05, 0.02, 0.01, -0.03]], dtype=torch.float32)
action = torch.tensor([[1]])
reward = torch.tensor([[1.0]])
next_state = torch.tensor([[0.07, 0.04, 0.005, -0.02]], dtype=torch.float32)
done = torch.tensor([[0.0]])  # not terminal.

# -------- Forward pass --------#
q_values = policy_net(state)
q_value = q_values.gather(1, action)  # fetch Q(s, a).

with torch.no_grad():
    next_qs = target_net(next_state)
    max_next_q = next_qs.max(1, keepdim=True)[0]

target = reward + 0.99 * max_next_q * (1 - done)

# -------- Loss & update --------#
loss = loss_fn(q_value, target)
print(f"Q-value: {q_value.item():.4f} | Target: {target.item():.4f} | Loss: {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Weights.
print("\nUpdating Done! New Q-value after one step:")
new_q = policy_net(state)
print(new_q)
