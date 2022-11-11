from collections import deque
from transition import Transition
from random import sample

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, state, new_state, action, reward):
        """Save a transition, and if full, removes the leftmost transition (oldest)"""
        self.memory.append(Transition(state, new_state, action, reward))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
