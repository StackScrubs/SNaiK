from collections import deque
from typing_extensions import Self
from typing import Any
import random
from enum import Enum

DIRECTIONS = [
    (0, 1), # Up
    (1, 0),  # Right
    (0,-1), # Down
    (-1,0) # Left
]
DIRECTION_NONE = -1

class GridCellType(Enum):
    SNAKE = 1
    APPLE = 2

class Grid:
    def __init__(self, size):
        self.size = size
        self.__cells = [None for _ in range(size*size)]
        self.__free_cells = len(self.__cells)

    def _pos_to_index(self, pos):
        return pos[0] + pos[1]*self.size

    def _index_to_pos(self, idx):
        return (idx % self.size, idx // self.size)
    
    def _move_cell(self, src_idx, dst_idx):
        if self.__cells[dst_idx] is not None:
            return self.__cells[dst_idx]
        self.__cells[dst_idx] = self.__cells[src_idx]
        self.__cells[src_idx] = None
        return None

    def _free_cell(self, idx):
        self.__cells[idx] = None
        self.__free_cells += 1

    def _new_cell(self, idx, val):
        if self.__cells[idx] is not None:
            raise Exception(f"cell {idx} is already occupied")
        cell = GridCell(self, idx, val)
        self.__cells[idx] = cell
        self.__free_cells -= 1
        return cell

    @property
    def cells(self):
        for i in range(len(self.__cells)):
            val = None
            if self.__cells[i] is not None:
                val = self.__cells[i].value
            yield (self._index_to_pos(i), val)

    def new_cell(self, pos, val):
        index = self._pos_to_index(pos)
        return self._new_cell(index, val)

    def find_free_cell(self):
        n = random.randint(0, self.__free_cells - 1)
        f = -1
        free_idx = -1
        for i in range(len(self.__cells)):
            if self.__cells[i] is None:
                f += 1
            if f == n:
                free_idx = i
                break

        return self._index_to_pos(free_idx)
        
class GridCell:
    def __init__(self: Self, grid: Grid, index: int, value: Any):
        self.__grid = grid
        self.__index = index
        self.__value = value

    @property
    def position(self):
        return self.__grid._index_to_pos(self.__index)

    def move(self, pos):
        dst_idx = self.__grid._pos_to_index(pos)
        collided = self.__grid._move_cell(self.__index, dst_idx)
        if collided is not None:
            src_pos = self._index_to_pos(self.__index)
            raise Exception(f"cannot move cell from {src_pos} to {pos}: destination is occupied by {self.cells[dst_idx]}")
        self.__index = dst_idx

    def try_move(self, pos) -> Self | None:
        dst_idx = self.__grid._pos_to_index(pos)
        collided = self.__grid._move_cell(self.__index, dst_idx)
        if collided is not None:
            return collided
        self.__index = dst_idx
        return None

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, val):
        self.__value = val
        
    def free(self):
        self.__grid._free_cell(self.__index)
        self.__grid = None
        self.index = None
        self.value = None

    def __str__(self):
        return f"GridCell({self.__value} @ {self.position})"


class SnakeState:
    def __init__(self, size):
        self.grid = Grid(size)
        self.snake = deque([self.grid.new_cell((1, 1), GridCellType.SNAKE)])
        self.alive = True
        self.apple = self.grid.new_cell(self.grid.find_free_cell(), GridCellType.APPLE)
        self.direction_index = 2 # implement random
        self.steps = 0

    # Snakes relative turn direction, converted to constant env direction
    def turn_left(self):
        self.direction_index = (self.direction_index - 1 + len(DIRECTIONS)) % len(DIRECTIONS)
    
    def turn_right(self):
        self.direction_index = (self.direction_index + 1) % len(DIRECTIONS)

    def update(self):
        if not self.alive:
            return

        ate = self._move_snake()

        if ate:
            self._grow_snake()

        self.steps += 1
        return ate

    def _move_snake(self) -> bool:
        if self.direction_index == DIRECTION_NONE: 
            return

        head = self.snake[0]
        new_head_pos = (
            head.position[0] + DIRECTIONS[self.direction_index][0],
            head.position[1] + DIRECTIONS[self.direction_index][1]
        )

        if new_head_pos[0] < 0 or new_head_pos[0] >= self.grid.size or new_head_pos[1] < 0 or new_head_pos[1] >= self.grid.size:
            self.alive = False
            return

        tail = self.snake.pop()

        for body_part in self.snake:
            if new_head_pos == body_part:
                self.alive = False

        ate = False
        hit = tail.try_move(new_head_pos)
        if hit is not None and hit.value == GridCellType.APPLE:
            self._respawn_apple()
            tail.move(new_head_pos)
            ate = True

        self.snake.appendleft(tail)
        return ate

    def _grow_snake(self):
        self.snake.append(self.snake[-1])

    def _respawn_apple(self):
        self.apple.move(self.grid.find_free_cell())

    def has_won(self):
        return len(self.snake) >= self.grid.size**2