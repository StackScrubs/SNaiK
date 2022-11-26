from __future__ import annotations
from snake_env import SnakeEnv
from dataclasses import dataclass
from functools import cached_property
from copy import copy

@dataclass(frozen=True)
class AgentContext:
    alpha: float
    gamma: float
    size: int
    seed: int | None
    
    @cached_property
    def env(self) -> SnakeEnv:
        return SnakeEnv(render_mode=None, seed=self.seed, size=self.size)

class Context:
    def __init__(self, alpha: float, gamma: float, size: int, episodes: int, seed: int | None, render: bool):
        self.render = render
        self.episodes = episodes
        
        self.__agent_context = AgentContext(alpha=alpha, gamma=gamma, size=size, seed=seed)
    
    @property
    def agent_context(self) -> AgentContext:
        return copy(self.__agent_context)
    
    @property
    def alpha(self) -> float:
        return self.__agent_context.alpha
    
    @property
    def gamma(self) -> float:
        return self.__agent_context.gamma
    
    @property
    def size(self) -> int:
        return self.__agent_context.size
    
    @property
    def seed(self) -> int:
        return self.__agent_context.seed
    
    @cached_property
    def env(self) -> SnakeEnv:
        return self.__agent_context.env

    @cached_property
    def render_env(self) -> SnakeEnv | None:
        if not self.render:
            return None
        return SnakeEnv(render_mode="human", seed=self.seed, size=self.size)

    @cached_property
    def info(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "grid_size": self.size,
        }
        
    def __getstate__(self):
        copy = self.__dict__.copy()
        for k in ("env", "render_env", "info"):
            if k in copy:
                del copy[k]
        return copy

    def __setstate__(self, state):
        self.__dict__.update(state)
