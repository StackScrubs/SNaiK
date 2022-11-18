from collections import deque
from transition import Transition
from random import shuffle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, new_state, action, reward):
        """Save a transition, and if full, removes the leftmost transition"""
        if len(self.memory) >= self.capacity:
            self.memory.pop()
        
        self.memory.append(Transition(state, new_state, action, reward))

    def sample(self, batch_size):
        self._shuffle()
        return self.memory[:batch_size]

    def _shuffle(self):
        shuffle(self.memory)

    def __len__(self):
        return len(self.memory)
