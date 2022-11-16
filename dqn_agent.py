from replay_memory import ReplayMemory
from gymnasium import Env
import random
from linear_dqn import LinDQN
import numpy as np

import torch

def dqn_main(env: Env, seed, epochs: int):
    BEGINNING_MEMORY_TRANSITIONS = 100

    agent = DQNAgent()

    # Create starting memory
    for i in range(BEGINNING_MEMORY_TRANSITIONS):
        action = random.choice([0,1,2]) #optimize is slow :-()
        new_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset(seed=seed)

        agent.memorize(state, new_state, action, reward)

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

def _tensorize(grid_size: int, observation) -> int:
    n_squares = grid_size*grid_size
    dvec = lambda v: v.x*grid_size + v.y
    
    apple_obs = observation["apple"]
    apple_obs = dvec(apple_obs) if not apple_obs is None else n_squares

    return torch.tensor([
        dvec(observation["head"]) * n_squares**3,
        dvec(observation["tail"]) * n_squares**2,
        apple_obs * n_squares,
        (observation["length"] - 1)
    ])

class DQNAgent:

    LEARNING_RATE = 0.001
    MEMORY_SIZE = 5000
    T = 50

    def __init__(self, env):
        self.env = env
        self.batch_size = 32
        self.explore_prob = 0.9
        self.explore_prob_end = 0.05
        self.explore_prob_decay = 200
        self.total_steps = 0

        self.q_network = LinDQN()
        self.target_network = LinDQN()

        self.memory_buffer = ReplayMemory(self.MEMORY_SIZE)
        self.state = 0
    
    # Store transition
    def memorize(self, state, new_state, action, reward): 
        self.memory_buffer.push(state, action, new_state, reward)
    
    # return an action
    def act(self, memory):
        epsilon = self.explore_prob_end \
                    + (self.explore_prob - self.explore_prob_end) \
                    * np.exp(-1 * self.total_steps / self.explore_prob_decay)

        self.total_steps += 1

        if random.random() < epsilon:
            # explore
            return random.choice([0,1,2])
        else:
            # exploit
            return memory.action

    def experience_replay(self):
        action = random.randint(0, 2)

        new_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self.env.reset() # seed

        self.memorize(self.state, _tensorize(8, new_state), action, reward)
        self.state = _tensorize(8, new_state)

    def train(self):
        self.q_network.train(True)
        self.target_network.train(False)

        self.q_network.init_layer_weights()
        self.target_network.load_state_dict(self.q_network.state_dict())

        #for i in range(self.batch_size):
         #   self.experience_replay()

        optimizer = torch.optim.Adam(self.q_network.parameters(), self.LEARNING_RATE)
        EPISODES = 10_000
        for i in range(EPISODES):
            for i in range(self.batch_size):
                self.experience_replay()

            memories = self.memory_buffer.sample(self.batch_size)

            for memory in memories:
                target_q_values = self.target_network.f(memory.new_state)
                target_q_value = np.max(target_q_values) + memory.reward

                self.q_network.loss(memory.state, memory.action, target_q_value).backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % self.T == 0:
                # copy q network's weights to target network every T step
                self.target_network.load_state_dict(self.q_network)



    # def experience_replay(self):
    #     memories = self.memory_buffer.sample(self.batch_size)

    #     q_value = -1
    #     action = None

    #     for memory in memories:
    #         new_q_value = self.q_network.f(memory)

    #         if new_q_value > q_value: # TODO: Implement some form of greedy algorithm
    #             q_value = new_q_value
    #             action = memory.action
                
    #     return action

    def _load_model():
        
        pass

    def _save_model():
        
        pass
