from copy import deepcopy
from replay_memory import ReplayMemory
from gymnasium import Env
import random
from conv_dqn import ConvolutionalDQN
import numpy as np
import torch
import pickle
from typing_extensions import Self

class DQNAgent:

    ALPHA = 0.01
    GAMMA = 0.9999
    MEMORY_SIZE = 50_000
    T = 32

    EXPLORE_PROB = 0.9
    EXPLORE_PROB_END = 0.05
    EXPLORE_PROB_DECAY = 200
    EPS_START = 0.9
    EPS_END = 0.05  # Petition to rename to this? Sice we use epsilon in get_action()
    EPS_DECAY = 200

    def __init__(self, grid_size):
        self.batch_size = 32
        self.total_steps = 0
        self.grid_size = grid_size

        self.q_network = ConvolutionalDQN(self.grid_size)
        self.target_network = ConvolutionalDQN(self.grid_size)

        self.memory_buffer = ReplayMemory(self.MEMORY_SIZE)
        self.state = None

        self.q_network.train(True)
        self.target_network.train(False)
        
        self.q_network.init_layers()
        self.copy_q_to_target()

        # SGD because NN is too small for Adam (???????)
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=self.ALPHA)

    @property
    def memory_sample(self):
        return self.memory_buffer.sample(self.batch_size)

    def memorize(self, state: torch.Tensor, new_state: torch.Tensor, action: int, reward: int): 
        self.memory_buffer.push(state, new_state, action, reward)

    def get_random_action(self):
        return random.randint(0, 2)

    def get_action(self) -> int:
        epsilon = self.EXPLORE_PROB_END \
                    + (self.EXPLORE_PROB - self.EXPLORE_PROB_END) \
                    * np.exp(-1 * self.total_steps / self.EXPLORE_PROB_DECAY)

        self.total_steps += 1

        if random.random() < epsilon:
            return self.get_random_action()
        else:
            action_q_vals = self.q_network.f(self.state)
            return torch.argmax(action_q_vals).item()

    def experience_replay(self, env, explore_only=False):
        action = self.get_action() if not explore_only else self.get_random_action()

        new_state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            new_state = env.reset()

        new_state = self.__tensorize_state(new_state)
        self.memorize(self.state, new_state, action, reward)
        self.state = new_state

    def experience_initial(self, env):
        self.state = self.__tensorize_state(env.reset())
        for _ in range(1024):
            self.experience_replay(env, explore_only=True)
        self.state = self.__tensorize_state(env.reset())

    def get_optimal_action(self, state):
        state = self.__tensorize_state(state)
        action_q_vals = self.q_network.f(state) 
        return torch.argmax(action_q_vals).item()

    def train_q_network(self):
        states, new_states, actions, rewards = self.memory_buffer.sample_batched(self.batch_size)
        
        # Reshape the states so they fit in the DQN model
        states = states.view(self.batch_size, 1, self.grid_size, self.grid_size)
        new_states = new_states.view(self.batch_size, 1, self.grid_size, self.grid_size)
        
        q_arrays = self.target_network.f(new_states)
        q_max, _ = torch.max(q_arrays, axis=1, keepdim=True)
        q_values = torch.multiply(q_max, self.GAMMA) + rewards.view(-1, 1)

        self.q_network.loss(states, actions, q_values).backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()

    def copy_q_to_target(self):
        self.target_network.load_state_dict(deepcopy(self.q_network.state_dict()))

    def __tensorize_state(self, state) -> torch.Tensor:
        return torch.tensor(state["grid"]).reshape(self.grid_size, self.grid_size).float() / 3

    def to_file(self, base_path = "."):
        from time import time
        pickle_dump = pickle.dumps(self)
        with open(f"{base_path}/dqn_model_{time()}.qbf", "wb") as f:
            f.write(pickle_dump)

    @staticmethod
    def from_file(file_path) -> Self:
        with open(file_path, "rb") as f:
            agent = pickle.loads(f.read())
            return agent
