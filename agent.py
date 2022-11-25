from __future__ import annotations
from random import randint
from typing import Any
from typing_extensions import Self
from snake_env import LazyDict, SnakeEnv
from qtable import QTable
from discretizer import Discretizer
from transition import Transition
from enum import Enum
from pickle import dumps, loads
from utils.context import AgentContext
import numpy as np

class AgentType(str, Enum):
    RANDOM = "random"
    QLEARNING = "qlearning"

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
        return self._env_update(action)
    
    def run_episode(self):
        return super().run_episode()


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
        
    def run_episode(self):
        return super().run_episode()
            
    @property
    def info(self) -> dict:
        return {
            **super().info,
            "discretizer": self.__discretizer.info,
        }
