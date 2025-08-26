import numpy as np
from typing import List, Tuple, Any

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: List[Tuple[Any, Any, float, Any, bool]] = []
        self.max_capacity = capacity
        self.index = 0
        self.size = 0

    def store(self, experience: Tuple[Any, Any, float, Any, bool]):
        if self.size < self.max_capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.index] = experience
            self.index = (self.index + 1) % self.max_capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return zip(*batch)

    def __len__(self):
        return self.size
