from random import randint
import torch

class ReplayMemory:
    def __init__(self, capacity, channels, grid_size):
        self.__cap = capacity
        self.__states = torch.zeros(capacity, channels, grid_size, grid_size)
        self.__new_states = torch.zeros(capacity, channels, grid_size, grid_size)
        self.__actions = torch.zeros(capacity, dtype=torch.long)
        self.__rewards = torch.zeros(capacity)
        self.__write_index = 0
        self.__len = 0

    def push(self, state, new_state, action, reward):
        """Save a transition, and if full, overwrites the leftmost transition"""
        self.__len = min(self.__len + 1, self.__cap)
        self.__states[self.__write_index] = state
        self.__new_states[self.__write_index] = new_state
        self.__actions[self.__write_index] = action
        self.__rewards[self.__write_index] = reward
        self.__write_index = (self.__write_index + 1) % self.__cap
    
    def random_indices(self, batch_size):
        s = set()
        while len(s) < batch_size:
            s.add(randint(0, self.__len - 1))
        return torch.tensor(list(s), dtype=torch.long)
    
    def sample_batched(self, batch_size):
        indices = self.random_indices(batch_size)
        states = torch.index_select(self.__states, 0, indices)
        new_states = torch.index_select(self.__new_states, 0, indices)
        actions = torch.index_select(self.__actions, 0, indices)
        rewards = torch.index_select(self.__rewards, 0, indices)
        return states, new_states, actions, rewards

    def __len__(self):
        return self.__len
    
    def __str__(self):
        return f"{self.__list[:self.__len]}"
