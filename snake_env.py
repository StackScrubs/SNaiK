import gymnasium as gym
from gymnasium import spaces
from snake_state import SnakeState, GridCellType

class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": [None, "human"],
        "render_fps": 0
    }

    def __init__(self, render_mode=None, seed=None, size=8):
        if render_mode not in self.metadata["render_modes"]:
            return

        self.state = SnakeState(size, seed)
        self.size = size
        self.window_size = 1024
        self.seed = seed

        # Target's location, Neo's location and length
        self.observation_space = spaces.Dict({
            "head": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "tail": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "apple": spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "length": spaces.Box(0, self.size, dtype=int)
        })

        # Action space for turning left or right, or remaining idle
        self.action_space = spaces.Discrete(3)

        self.render_mode = render_mode
        self.screen = None
        self.window = None
        self.clock = None
        self.death_counter = 0

    # Snakes relative turn direction, converted to constant env direction
    def _turn_snake(self, action):
        if action == 1:
            self.state.turn_left()
        elif action == 2:
            self.state.turn_right()

    def step(self, action):
        self._turn_snake(action)

        ate = self.state.update()
        won = self.state.has_won
        reward = 0

        observation = self._get_obs()
        
        if ate:
            reward = 1 / self.state.steps
        else:
            reward = 1 / self.state.dist_to_apple
        if won:
            reward = 5
        terminated = won
        truncated = not self.state.is_alive
        info = None

        if self.render_mode == "human":
            self._render()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "head": self.state.head_position,
            "tail": self.state.tail_position,
            "apple": self.state.apple_position,
            "length": self.state.snake_length
        }

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed
        self.state = SnakeState(self.size, self.seed)
        self.death_counter += 1

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

        self.clock.tick(self.metadata["render_fps"])
        pygame.display.update()
        pygame.event.pump()

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
