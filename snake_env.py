from typing import Any
import gymnasium as gym
from gymnasium import spaces
from snake_state import SnakeState, GridCellType
from collections.abc import Mapping
import datetime
import numpy as np

class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self.__dict = dict(*args, **kw)
        self.__ldict = {}
    
    def __init_lazy(self, key):
        if key not in self.__ldict and key in self.__dict:
            self.__ldict[key] = self.__dict.pop(key)()
            
    def __getitem__(self, key):
        self.__init_lazy(key)
        return self.__ldict[key]
        
    def __iter__(self):
        for key in self.__ldict.keys():
            yield self[key]
        keys = list(self.__dict.keys())
        for key in keys:
            yield self[key]
            
    def __len__(self):
        return len(self.__dict) + len(self.__ldict)
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def __getstate__(self):
        keys = list(self.__dict.keys())
        for key in keys:
            self.__init_lazy(key)
        return self.__dict__
        

class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": [None, "human"],
        "render_fps": 60
    }

    def __init__(self, render_mode=None, seed=None, size=8):
        if render_mode not in self.metadata["render_modes"]:
            return

        self.state = SnakeState(size, seed)
        self.size = size
        self.window_size = 1024
        self.seed = seed
        self.steps = 0
        self.score = 0

        # Target's location, Neo's location and length
        self.observation_space = spaces.Dict({
            "head": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "direction": spaces.Discrete(4),
            "tail": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "apple": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "length": spaces.Box(0, self.size, dtype=int),
            "grid": spaces.Box(0, self.size, shape=(2, ), dtype=int),
            "collidables": spaces.Box(0, self.size**2, shape=(1, ), dtype=int),
        })

        # Action space for turning left or right, or remaining idle
        self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.screen = None
        self.window = None
        self.clock = None
        self.last_render_ms = None
        self.death_counter = 0

    # Snakes relative turn direction, converted to constant env direction
    def _turn_snake(self, action):
        if action == 1:
            self.state.turn_left()
        elif action == 2:
            self.state.turn_right()

    def step(self, action):
        self._turn_snake(action)
        dist_to_apple = self.state.head_position.manhattan_dist(self.state.apple_position)
        
        ate = self.state.update()
        won = self.state.has_won
        observation = self._get_obs()
        
        self.steps += 1
        reward = 0
        if ate:
            self.score += 1
            reward = 1 / self.steps
            self.steps = 0
        else:
            new_dist_to_apple = self.state.head_position.manhattan_dist(self.state.apple_position)
            reward = dist_to_apple - new_dist_to_apple
        if won:
            reward = 50
        if not self.state.is_alive:
            reward = -50
            
        terminated = won
        truncated = not self.state.is_alive
        info = self.score

        self._render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        get_grid = lambda: np.fromiter(
            (c[1] if c[1] is not None else 0. for c in self.state.grid_cells), 
            np.uint8
        ).reshape(self.size, self.size)
        
        return LazyDict({
            "head": lambda: self.state.head_position,
            "direction": lambda: self.state.direction,
            "tail": lambda: self.state.tail_position,
            "apple": lambda: self.state.apple_position,
            "length": lambda: self.state.snake_length,
            "grid": get_grid,
            "collidables": lambda: self.state.collidables,
        })
    
    def reset(self, seed: Any = None):
        if seed is None:
            seed = self.seed
        self.state = SnakeState(self.size, seed)
        self.score = 0
        self.death_counter += 1
        self._render()
        return self._get_obs()

    @property
    def can_render(self):
        import pygame
        return (
            self.last_render_ms is None or 
            (pygame.time.get_ticks() - self.last_render_ms) > (1 / self.metadata["render_fps"])*1000
        )

    def _render(self):
        if self.render_mode != "human":
            return

        import pygame

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.window_size, ) * 2)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        square_size = self.window_size / self.size
        surface = pygame.Surface((self.window_size,) * 2)
        BG_COLOR = 205
        surface.fill((BG_COLOR,) * 3)

        for pos, v in self.state.grid_cells:
            l = square_size * pos.x
            t = square_size * pos.y
            square_color, is_bordered = self._get_square_display(v)
            rect = pygame.Rect(l, t, square_size, square_size)
            pygame.draw.rect(surface, square_color, rect, is_bordered)

        self.screen.blit(surface, (0, 0))

        font = pygame.font.SysFont(None, 24)
        img = font.render(f"Deaths: {self.death_counter}", True, 255)
        self.screen.blit(img, (20, 20))
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        self.last_render_ms = pygame.time.get_ticks()

    def _get_square_display(self, cell_type):
        if cell_type is None:
            return ((55,) * 3, True)
        elif cell_type == GridCellType.SNAKE_BODY:
            return ((0, 64, 255), False)
        elif cell_type == GridCellType.SNAKE_HEAD:
            return ((0, 255, 64), False)
        elif cell_type == GridCellType.APPLE:
            return ((255, 0, 64), False)

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
