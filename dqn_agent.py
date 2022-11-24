import random
import numpy as np
import torch
import pickle
from copy import deepcopy
from replay_memory import ReplayMemory
from conv_dqn import ConvolutionalDQN
from linear_dqn import LinearDQN
from typing_extensions import Self

class DQNAgent:
    
    #LinearDQN vals
    #ALPHA = 0.001
    #GAMMA = 0.9

    #ConvDQN vals
    ALPHA = 0.001
    GAMMA = 0.9
    MEMORY_SIZE = 50_000
    T = 50

    EPSILON_START = 0.9
    EPSILON_END = 0.05
    EPSILON_DECAY = 125 # 200 a wee bit extreme, needs tweaking
    
    CHANNELS = 3

    def __init__(self, grid_size):
        self.batch_size = 32
        self.total_steps = 0
        self.grid_size = grid_size

        self.policy_net = LinearDQN(self.grid_size)
        self.target_net = LinearDQN(self.grid_size)

        self.replay_memory = ReplayMemory(self.MEMORY_SIZE)
        self.state = None

        self.policy_net.train(True)
        self.target_net.train(False)
        
        self.policy_net.init_layers()
        self.copy_q_to_target()

        # SGD because NN is too small for Adam (???????)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.ALPHA)

    def memorize(self, state: torch.Tensor, new_state: torch.Tensor, action: int, reward: int): 
        self.replay_memory.push(state, new_state, action, reward)

    def get_random_action(self):
        return random.randint(0, 2)

    def get_action(self) -> int:
        epsilon = self.EPSILON_END \
                    + (self.EPSILON_START - self.EPSILON_END) \
                    * np.exp(-1 * self.total_steps / self.EPSILON_DECAY)

        self.total_steps += 1

        if random.random() < epsilon:
            #print(f"bro doing the rng {self.total_steps}")
            return self.get_random_action()
        else:
            action_q_vals = None
            with torch.no_grad():
                action_q_vals = self.policy_net(self.state.view(1, self.CHANNELS, self.grid_size, self.grid_size))
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
        with torch.no_grad():
            action_q_vals = self.policy_net(state.view(1, self.CHANNELS, self.grid_size, self.grid_size)) 
        return torch.argmax(action_q_vals).item()

    def train_q_network(self):
        states, new_states, actions, rewards = self.replay_memory.sample_batched(self.batch_size)
        
        # Reshape the states so they fit in the DQN model
        states = states.view(self.batch_size, self.CHANNELS, self.grid_size, self.grid_size)
        new_states = new_states.view(self.batch_size, self.CHANNELS, self.grid_size, self.grid_size)
        
        q_arrays = None
        with torch.no_grad():
            q_arrays = self.target_net(new_states)
        q_max, _ = torch.max(q_arrays, axis=1, keepdim=True)
        q_values = torch.multiply(q_max, self.GAMMA) + rewards.view(-1, 1)

        self.optimizer.zero_grad()
        self.policy_net.loss(states, actions, q_values).backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

    def copy_q_to_target(self):
        self.target_net.load_state_dict(deepcopy(self.policy_net.state_dict()))

    def __tensorize_state(self, state) -> torch.Tensor:
        t = torch.tensor(state["grid"], dtype=torch.uint8)
        s = torch.zeros((self.CHANNELS, self.grid_size, self.grid_size))
        for i in range(0, self.CHANNELS):
            s[i] = torch.where(t == i + 1, t // (i + 1), 0)
        return s

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
