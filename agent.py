from __future__ import annotations
from random import randint
from typing import Any
from snake_env import LazyDict, SnakeEnv
from qtable import QTable
from discretizer import Discretizer
from transition import Transition
from enum import Enum
from utils.context import AgentContext
from copy import deepcopy
from replay_memory import ReplayMemory
from dqn import DQN
import torch
import numpy as np

class AgentType(str, Enum):
    RANDOM = "random"
    QLEARNING = "qlearning"
    DQN = "dqn"

class Agent:
    def __init__(self, ctx: AgentContext):
        self.env = ctx.env
        self.observation = ctx.env.reset()
        self.reward = 0
    
    def get_optimal_action(self, observation) -> int:
        """Returns the optimal action for the provided observation"""
        pass
    
    def initialize(self):
        """Initialize the agent"""
        pass
        
    def update(self) -> tuple[LazyDict, float | int, bool, bool, Any]:
        """Runs a single training step"""
        pass
    
    def _env_update(self, action: int) -> tuple[LazyDict, float | int, bool, bool, Any]:
        return self.env.step(action)
            
    def run_episode(self):
        while True:
            self.observation, self.reward, terminated, truncated, info = self.update()
            if terminated or truncated:
                self.observation = self.env.reset()
                return info
            
    @property
    def info(self) -> dict:
        return {
            "type": self.TYPE,
        }

class RenderingAgentDecorator(Agent):
    def __init__(self, render_env: SnakeEnv, agent: Agent):
        self.agent = agent
        self.__render_env = render_env
        self.__observation = agent.observation
            
    def __try_render_once(self):
        if not self.__render_env.can_render:
            return
        
        self.__render_env.death_counter = self.agent.env.death_counter
        action = self.agent.get_optimal_action(self.__observation)
        self.__observation, _, terminated, truncated, _ = self.__render_env.step(action)
        
        if terminated or truncated:
            self.__observation = self.__render_env.reset()
    
    def get_optimal_action(self, observation) -> int:
        return self.agent.get_optimal_action(observation)
    
    def initialize(self):
        return self.agent.initialize()
        
    def update(self):
        self.__try_render_once()
        return self.agent.update()
    
    def _env_update(self, action: int):
        return self.agent._env_update(action)

    @property
    def env(self) -> SnakeEnv:
        return self.agent.env
    
    @property
    def observation(self) -> tuple[LazyDict, float | int, bool, bool, Any]:
        return self.agent.observation
    
    @observation.setter
    def observation(self, value: tuple[LazyDict, float | int, bool, bool, Any]):
        self.agent.observation = value
    
    @property
    def reward(self) -> float:
        return self.agent.reward
    
    @reward.setter
    def reward(self, value: float):
        self.agent.reward = value
    
    @property
    def info(self) -> dict:
        return self.agent.info


class RandomAgent(Agent):
    TYPE = AgentType.RANDOM
    
    def __init__(self, ctx: AgentContext):
        super().__init__(ctx)
    
    def __get_random_action(self):
        return randint(0, 2)
    
    def get_optimal_action(self, observation) -> int:
        return self.__get_random_action()
    
    def update(self):
        action = self.__get_random_action()
        self.__step += 1
        return self._env_update(action)
    
    
class QLearningAgent(Agent):
    TYPE = AgentType.QLEARNING
    
    def __init__(self, ctx: AgentContext, discretizer: Discretizer):
        super().__init__(ctx)
        self.__action_space_len = 3
        self.__state = None
        self.__action = None
        self.__step = 0
        self.__discretizer = discretizer
        self.__q = QTable(ctx.alpha, ctx.gamma, self.__action_space_len, discretizer.state_space_len)

    def get_optimal_action(self, observation):
        return self.__q.policy(self.__discretizer.discretize(observation))

    def update(self):
        new_state = self.__discretizer.discretize(self.observation)
        self.__q.update_entry(Transition(self.__state, new_state, self.__action, self.reward))
        action = self.__get_action(new_state)
        self.__state = new_state
        self.__action = action
        self.__step += 1
        return self._env_update(action)

    def __get_action(self, new_state):
        if np.random.random() < QTable.get_epsilon(self.__step):
            return np.random.randint(self.__action_space_len - 1)
        else:
            return self.__q.policy(new_state)

    @property
    def info(self) -> dict:
        return {
            **super().info,
            "discretizer": self.__discretizer.info,
        }


class DQNAgent(Agent):
    TYPE = AgentType.DQN
    
    def __init__(self, ctx: AgentContext, nn: DQN):
        super().__init__(ctx)
        self.nn = nn
        self.steps = 0
        self.grid_size = ctx.size
        self.alpha = ctx.alpha
        self.gamma = ctx.gamma

        self.channels = 3
        self.memory_size = 100_000
        self.batch_size = 256
        self.T = 1100
        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 500

        self.policy_net = self.nn
        self.target_net = deepcopy(self.nn)

        self.replay_memory = ReplayMemory(self.memory_size, self.channels, self.grid_size)
        self.tensorized_obs = self.__tensorize_observation(ctx.env.reset())

        self.policy_net.train(True)
        self.target_net.train(False)
        
        self.policy_net.init_layers()
        self.__copy_q_to_target()

        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.alpha)

    def get_optimal_action(self, observation) -> int:
        observation = self.__tensorize_observation(observation)
        with torch.no_grad():
            action_q_vals = self.policy_net(observation.view(1, self.channels, self.grid_size, self.grid_size)) 
        return torch.argmax(action_q_vals).item()

    def initialize(self):
        self.__experience_initial()

    def update(self):
        observation, reward, terminated, truncated, info = self.__experience_replay()
        
        states, new_states, actions, rewards = self.replay_memory.sample_batched(self.batch_size)
        
        # Reshape the states so they fit in the DQN model
        states = states.view(self.batch_size, self.channels, self.grid_size, self.grid_size)
        new_states = new_states.view(self.batch_size, self.channels, self.grid_size, self.grid_size)
        
        q_arrays = None
        with torch.no_grad():
            q_arrays = self.target_net(new_states)
        q_max, _ = torch.max(q_arrays, axis=1, keepdim=True)
        q_values = torch.multiply(q_max, self.gamma) + rewards.view(-1, 1)

        self.optimizer.zero_grad()
        self.policy_net.loss(states, actions, q_values).backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp(-1, 1)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.T == 0:
            self.__copy_q_to_target()
            
        return observation, reward, terminated, truncated, info

    def __memorize(self, state: torch.Tensor, new_state: torch.Tensor, action: int, reward: int): 
        self.replay_memory.push(state, new_state, action, reward)

    def __get_random_action(self):
        return randint(0, 2)

    def __get_action(self) -> int:
        epsilon = self.epsilon_end \
                    + (self.epsilon_start - self.epsilon_end) \
                    * np.exp(-1 * self.steps / self.epsilon_decay)
                    
        if np.random.random() < epsilon:
            return self.__get_random_action()
        else:
            action_q_vals = None
            with torch.no_grad():
                action_q_vals = self.policy_net(self.tensorized_obs.view(1, self.channels, self.grid_size, self.grid_size))
            return torch.argmax(action_q_vals).item()

    def __experience_replay(self, explore_only=False):
        action = self.__get_action() if not explore_only else self.__get_random_action()

        observation, reward, terminated, truncated, info = self._env_update(action)

        new_observation = self.__tensorize_observation(observation)
        self.__memorize(self.tensorized_obs, new_observation, action, reward)
        self.tensorized_obs = new_observation
        
        return observation, reward, terminated, truncated, info

    def __experience_initial(self):
        for _ in range(20_000):
            self.__experience_replay(explore_only=True)
        self.tensorized_obs = self.__tensorize_observation(self.env.reset())

    def __copy_q_to_target(self):
        self.target_net.load_state_dict(deepcopy(self.policy_net.state_dict()))

    def __tensorize_observation(self, observation) -> torch.Tensor:
        t = torch.tensor(observation["grid"], dtype=torch.uint8)
        s = torch.zeros((self.channels, self.grid_size, self.grid_size))
        for i in range(0, self.channels):
            s[i] = torch.where(t == i + 1, t // (i + 1), 0)
        return s

    @property
    def info(self) -> dict:
        return {
            **super().info,
            "nn": self.nn.info,
        }
