import gymnasium as gym
from gymnasium import spaces
import random
from collections import deque

DIRECTIONS = [
    (0, 1), # Up
    (1, 0),  # Right
    (0,-1), # Down
    (-1,0) # Left
]
DIRECTION_NONE = -1

class SnakeEnvState:
    def __init__(self, startPos):
        self.snake = deque([startPos])
        self.alive = True

class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 8
    }

    def __init__(self, render_mode=None, size=8):
        if render_mode not in self.metadata["render_modes"]:
            return

        self.state = SnakeEnvState((1,1))

        self.size = size
        self.window_size = 1024
                
        # Target's location and Neo's location
        self.observation_space = spaces.Dict({
            "head": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "apple": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            #"head_direction": spaces
        })
        
        # Action space for turning left or right, or remaining idle
        self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.screen = None
        self.window = None
        self.clock = None

        self.apple = (self.size /2, ) * 2
        self.direction_index = 2 # implement random

    # Snakes relative turn direction, converted to constant env direction
    def _turn(self, action):
        if action == 1: # LEFT
            self.direction_index = (self.direction_index - 1 + len(DIRECTIONS)) % len(DIRECTIONS)
        elif action == 2: #RIGHT
            self.direction_index = (self.direction_index + 1) % len(DIRECTIONS)

    def _move_snake(self):
        if self.direction_index == DIRECTION_NONE: 
            return

        head = self.state.snake[0]
        next_head = (
            head[0] + DIRECTIONS[self.direction_index][0],
            head[1] + DIRECTIONS[self.direction_index][1]
        )

        if next_head[0] < 0 or next_head[0] >= self.size or next_head[1] < 0 or next_head[1] >= self.size:
            self.state.alive = False
            return

        self.state.snake.pop()

        for body_part in self.state.snake:
            if next_head == body_part:
                self.state.alive = False

        self.state.snake.appendleft(next_head)

    def step(self, action):
        self._snake_step(action)

        observation = self._get_obs()
        reward = len(self.state.snake)
        terminated = self._has_won()
        truncated = not self.state.alive
        info = None

        if self.render_mode == 'human':
            self._render()

        return observation, reward, terminated, truncated, info

    def _grow_snake(self):
        self.state.snake.append(self.state.snake[-1])

    def _spawn_apple(self):
        # Naive implementation :haftingchad:
        # TODO: Profile and eventually improve

        size = self.size - (len(self.state.snake))
        apple_x, apple_y = self.state.snake[0]
        while (apple_x, apple_y) in self.state.snake:
            apple_x = random.randint(0, size)
            apple_y = random.randint(0, size)
        
        self.apple = (apple_x, apple_y)

    def _hit_apple(self):
        return self.state.snake[0] == self.apple

    def _has_won(self):
        return len(self.state.snake) >= self.size**2

    def _snake_step(self, action):
        if not self.state.alive:
            return
        
        self._turn(action)
        self._move_snake()

        if self._hit_apple():
            self._grow_snake()
            self._spawn_apple()

    def _get_obs(self):
        return {
            "head": self.state.snake[0],
            "apple": self.apple
        }

        
    def reset(self):
        self.state = SnakeEnvState(self.state.snake[0])

    def _render(self):
        if self.render_mode is None:
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

        l, t = 0, 0
        for x in range(self.size):
            for y in range(self.size):
                l = square_size * x
                t = square_size * y
                square_color, is_bordered = self._get_square_display((x, y))
                rect = pygame.Rect(l, t, square_size, square_size)
                pygame.draw.rect(surface, square_color, rect, is_bordered)

        self.screen.blit(surface, (0, 0))
        
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.update()
        pygame.event.pump()

    def _get_square_display(self, pos):
        if pos in self.state.snake:
            if pos == self.state.snake[0]:
                return ((0, 255, 64), False)
            else:
                return ((0, 64, 255), False)
        elif pos == self.apple:
            return ((255, 0, 64), False)
        else:
            return ((55,) * 3, True)

    def close(self):
        if self.window is not None:
            import pygame
            
            pygame.display.quit()
            pygame.quit()