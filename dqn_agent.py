from replay_memory import ReplayMemory
from gymnasium import Env
import random
from linear_dqn import LinDQN
import numpy as np

def dqn_main(env: Env, seed, epochs: int):
    BEGINING_MEMORY_TRANSITIONS = 100

    agent = DQNAgent()

    # Create starting memory
    for i in range(BEGINING_MEMORY_TRANSITIONS):
        action = random.choice([0,1,2]) #optimize is slow :-()
        new_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset(seed=seed)

        agent.memorize(state, action, new_state, reward)

        state = new_state

    # Create initially randomized QNN 

    # Copy QNN to TNN

    # Main loop
    for i in range(epochs):

        # Experience a replay
        greedy_action = agent.experience_replay()
        new_state, reward, terminated, truncated, info = env.step(greedy_action)
        agent.memorize(state, action, new_state, reward)

        # Train DQN
        
        pass

    pass

class DQNAgent():
    def __init__(self, action_space, state_space):
        self.batch_size = 32
        self.exploration_probability = 1.0
        self.exploration_probability_decay = 0.005

        state_sz = 4
        self.q_network = LinDQN(state_sz)
        self.target_network = LinDQN(state_sz)
        
        self.memory_buffer = ReplayMemory(5000)
    
    # Store transition
    def memorize(self, state, action, next_state, reward): 
        self.memory_buffer.push(state, action, next_state, reward)
    
    # return an action
    def act(): 
        return random.choice([0,1,2])

    def experience_replay(self):
        memories = self.memory_buffer.sample(self.batch_size)

        q_value = -1
        action = None

        for memory in memories:
            new_q_value = self.q_network.f(memory)

            if new_q_value > q_value: # TODO: Implement some form of greedy algorithm
                q_value = new_q_value
                action = memory.action
                
        return action

    def _load_model():
        
        pass

    def _save_model():
        
        pass
