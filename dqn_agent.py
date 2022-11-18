from copy import deepcopy
from replay_memory import ReplayMemory
from gymnasium import Env
import random
from linear_dqn import LinDQN
import numpy as np
import torch
import pickle
from typing_extensions import Self

class DQNAgent:

    ALPHA = 0.01
    GAMMA = 0.9
    MEMORY_SIZE = 50_000
    T = 32

    EXPLORE_PROB = 0.9
    EXPLORE_PROB_END = 0.05
    EXPLORE_PROB_DECAY = 200
    EPS_START = 0.9
    EPS_END = 0.05  # Petition to rename to this? Sice we use epsilon in get_action()
    EPS_DECAY = 200

    def __init__(self, env_size):
        self.batch_size = 64
        self.total_steps = 0
        self.env_size = env_size

        self.q_network = LinDQN()
        self.target_network = LinDQN()

        self.memory_buffer = ReplayMemory(self.MEMORY_SIZE)
        self.state = torch.tensor([0., 0., 0., 0.])

        self.q_network.train(True)
        self.target_network.train(False)
        
        self.q_network.init_layers()
        self.copy_q_to_target()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.ALPHA)

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
            env.reset()

        new_state = DQNAgent.__tensorize_state(self.env_size, new_state)
        self.memorize(self.state, new_state, action, reward)
        self.state = new_state

    def experience_initial(self, env):
        for _ in range(self.batch_size):
            self.experience_replay(env, explore_only=True)

    def get_optimal_action(self, state):
        state = DQNAgent.__tensorize_state(self.env_size, state)
        action_q_vals = self.q_network.f(state)
        return torch.argmax(action_q_vals).item()

    def train_q_network(self, memories):
        for memory in memories:
            target_q_values = self.target_network.f(memory.new_state)
            target_q_value = self.GAMMA * torch.max(target_q_values) + memory.reward

            self.q_network.loss(memory.state, memory.action, target_q_value).backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

    def copy_q_to_target(self):
        self.target_network.load_state_dict(deepcopy(self.q_network.state_dict()))

    @staticmethod
    def __tensorize_state(grid_size: int, state) -> torch.Tensor:
        n_squares = grid_size*grid_size
        dvec = lambda v: v.x*grid_size + v.y
        
        apple_obs = state["apple"]
        apple_obs = dvec(apple_obs) if apple_obs is not None else n_squares
        
        return torch.tensor([
            dvec(state["head"]),
            dvec(state["tail"]),
            apple_obs,
            state["length"]
        ]) / n_squares

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
