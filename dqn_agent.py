from replay_memory import ReplayMemory
from gymnasium import Env
import random
from linear_dqn import LinDQN
import numpy as np
import torch
import pickle
from typing_extensions import Self
from tqdm import tqdm

class DQNAgent:

    ALPHA = 0.9
    GAMMA = 0.1
    MEMORY_SIZE = 5_000
    T = 100

    EXPLORE_PROB = 0.9
    EXPLORE_PROB_END = 0.05
    EXPLORE_PROB_DECAY = 200

    def __init__(self, env: Env):
        self.env = env
        self.batch_size = 10
        self.total_steps = 0

        self.q_network = LinDQN()
        self.target_network = LinDQN()

        self.memory_buffer = ReplayMemory(self.MEMORY_SIZE)
        self.state = torch.tensor([0., 0., 0., 0.])
    
    def set_env(self, env):
        self.env = env

    def memorize(self, state: torch.Tensor, new_state: torch.Tensor, action: int, reward: int): 
        self.memory_buffer.push(state, new_state, action, reward)

    def act(self, explore_only = False) -> int:
        epsilon = self.EXPLORE_PROB_END \
                    + (self.EXPLORE_PROB - self.EXPLORE_PROB_END) \
                    * np.exp(-1 * self.total_steps / self.EXPLORE_PROB_DECAY)

        if not explore_only:
            self.total_steps += 1

        if explore_only or random.random() < epsilon:
            return random.randint(0, 2)
        else:
            action_q_vals = self.q_network.f(self.state)
            return torch.argmax(action_q_vals).item()

    def experience_replay(self, explore_only=False):
        action = self.act(explore_only)

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.env.reset()

        new_state = DQNAgent.__tensorize_state(self.env.size, new_state)
        self.memorize(self.state, new_state, action, reward)
        self.state = new_state

    def train(self):
        self.q_network.train(True)
        self.target_network.train(False)

        self.q_network.init_layers()
        self.copy_q_to_target()

        for _ in range(self.batch_size):
            self.experience_replay(explore_only=True)

        optimizer = torch.optim.Adam(self.q_network.parameters(), self.ALPHA)
        EPISODES = 10_000
        for i in tqdm(range(EPISODES)):
            self.experience_replay()

            memories = self.memory_buffer.sample(self.batch_size)

            for memory in memories:
                target_q_values = self.target_network.f(memory.new_state)
                target_q_value = self.GAMMA * torch.max(target_q_values) + memory.reward

                self.q_network.loss(memory.state, memory.action, target_q_value).backward()

            optimizer.step()
            optimizer.zero_grad()

            if i % self.T == 0:
                self.copy_q_to_target()

    def copy_q_to_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    @staticmethod
    def __tensorize_state(grid_size: int, state) -> torch.Tensor:
        n_squares = grid_size*grid_size
        dvec = lambda v: v.x*grid_size + v.y
        
        apple_obs = state["apple"]
        apple_obs = dvec(apple_obs) if apple_obs is not None else n_squares

        return torch.tensor([
            float(dvec(state["head"])),
            float(dvec(state["tail"])),
            float(apple_obs),
            float(state["length"])
        ])

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
