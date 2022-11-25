from transition import Transition
from random import shuffle, sample
from collections import deque
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.__cap = capacity
        self.__list = list(None for _ in range(capacity))
        self.__write_index = 0
        self.__len = 0

    def push(self, state, new_state, action, reward):
        """Save a transition, and if full, removes the leftmost transition"""
        self.__len = min(self.__len + 1, self.__cap)
        self.__list[self.__write_index] = Transition(state, new_state, action, reward)
        self.__write_index = (self.__write_index + 1) % self.__cap
    
    def sample_batched(self, batch_size):
        s = sample(self.__list[:self.__len], batch_size)
        states = torch.zeros(batch_size, *(first.state.shape))
        new_states = torch.zeros(batch_size, *(first.state.shape))
        actions = torch.zeros(batch_size, dtype=torch.long)
        rewards = torch.zeros(batch_size)
        
        for (state, new_state, action, reward, transition) in zip(states, new_states, actions, rewards, s):
            state.copy_(transition.state)
            new_state.copy_(transition.new_state)
            action.copy_(transition.action)
            reward.copy_(transition.reward)
            
        return states, new_states, actions, rewards

    def __len__(self):
        return self.__len
    
    def __str__(self):
        return f"{self.__list[:self.__len]}"
