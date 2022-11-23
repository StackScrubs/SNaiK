from transition import Transition
from random import shuffle
import torch

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
    
    def sample_batched(self, batch_size):
        sample = self.sample(batch_size)

        states = torch.zeros(batch_size, *(sample[0].state.shape))
        new_states = torch.zeros(batch_size, *(sample[0].state.shape))
        actions = torch.zeros(batch_size, dtype=torch.long)
        rewards = torch.zeros(batch_size)
        
        for i, transition in enumerate(sample):
            states[i] = transition.state
            new_states[i] = transition.new_state
            actions[i] = transition.action
            rewards[i] = transition.reward
            
        return states, new_states, actions, rewards
            

    def _shuffle(self):
        shuffle(self.memory)

    def __len__(self):
        return len(self.memory)
    
    def __str__(self):
        return f"{self.memory}"
