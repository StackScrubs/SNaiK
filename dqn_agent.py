from q_table import QTable
from replay_memory import ReplayMemory
import random


class DQNAgent:
    def __init__(self, action_space, state_space):
        self.batch_size = 32
        self.exploration_probability = 1.0
        self.exploration_probability_decay = 0.005
        
        self.q_table = QTable(action_space, state_space)
        self.memory_buffer = ReplayMemory(5000)
    
    # Store transition
    def memorize(self, state, action, next_state, reward): 
        self.memory_buffer.push(state, action, next_state, reward)
    
    # return an action
    def act(): 
        return random.choice([0,1,2])

    #Updated q_table based on a batch of transitions
    def replay(self): 
        memories = self.memory_buffer.sample(self.batch_size)
        
        pass

    def _load_model():
        
        pass

    def _save_model():
        
        pass
