from collections import deque
from typing_extensions import Self
from typing import Any, Tuple
import random
from enum import Enum

DIRECTIONS = [
    (0, 1), # Up
    (1, 0),  # Right
    (0,-1), # Down
    (-1,0) # Left
]

class Random:
    def __init__(self, seed):
        old_state = random.getstate()
        if seed is not None:
            random.seed(seed)
        self.state = random.getstate()
        random.setstate(old_state)

    def randint(self, a: int, b: int):
        old_state = random.getstate()
        random.setstate(self.state)
        val = random.randint(a, b)
        self.state = random.getstate()
        random.setstate(old_state)
        return val
    
class GridCellType(Enum):
    SNAKE = 1
    APPLE = 2
    HEAD = 3

class Grid:
    def __init__(self, size, random):
        self.size = size
        self.__cells = [None for _ in range(size*size)]
        self.__free_cells = len(self.__cells)
        self.__random = random

    def _pos_to_index(self, pos):
        return pos[0] + pos[1]*self.size

    def _index_to_pos(self, idx):
        return (idx % self.size, idx // self.size)
    
    def _move_cell(self, src_idx, dst_idx):
        if src_idx == dst_idx:
            return
        if self.__cells[dst_idx] is not None:
            src_pos = self._index_to_pos(src_idx)
            dst_pos = self._index_to_pos(dst_idx)
            raise Exception(f"cannot move cell from {src_pos} to {dst_pos}: destination is occupied by {self.__cells[dst_idx]}")
        self.__cells[dst_idx] = self.__cells[src_idx]
        self.__cells[src_idx] = None
        return

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

    def get_cell(self, pos):
        index = self._pos_to_index(pos)
        return self.__cells[index]

    def new_cell(self, pos, val):
        index = self._pos_to_index(pos)
        return self._new_cell(index, val)

    def find_free_cell(self):
        n = self.__random.randint(0, self.__free_cells - 1)
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
        self.__grid._move_cell(self.__index, dst_idx)
        self.__index = dst_idx

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
    def __init__(self, size, seed):
        self.__random = Random(seed)
        self.__grid = Grid(size, self.__random)
        self.__snake = deque([self.__grid.new_cell(self.__grid.find_free_cell(), GridCellType.SNAKE)])
        self.__alive = True
        self.__apple = self.__grid.new_cell(self.__grid.find_free_cell(), GridCellType.APPLE)
        self.__direction_index = self.__random.randint(0, len(DIRECTIONS) - 1)
        self.__steps = 0
        self.__to_grow = 0

    @property
    def alive(self) -> bool:
        return self.__alive

    @property
    def steps(self) -> int:
        return self.__steps

    @property
    def head_position(self) -> Tuple[int, int]:
        return self.__snake[0].position

    @property
    def tail_position(self) -> Tuple[int, int]:
        return self.__snake[-1].position

    @property
    def apple_position(self) -> Tuple[int, int]:
        return self.__apple.position

    @property
    def snake_length(self) -> int:
        return len(self.__snake) + self.__to_grow

    @property
    def grid_cells(self):
        return self.__grid.cells

    # Snakes relative turn direction, converted to constant env direction
    def turn_left(self):
        self.__direction_index = (self.__direction_index - 1 + len(DIRECTIONS)) % len(DIRECTIONS)
    
    def turn_right(self):
        self.__direction_index = (self.__direction_index + 1) % len(DIRECTIONS)

    def update(self):
        if not self.__alive:
            return False

        ate = self._snake_step()

        if ate:
            self._grow_snake()

        self.__steps += 1
        return ate

    def _snake_step(self) -> bool:
        head = self.__snake[0]
        new_head_pos = (
            head.position[0] + DIRECTIONS[self.__direction_index][0],
            head.position[1] + DIRECTIONS[self.__direction_index][1]
        )

        if new_head_pos[0] < 0 or new_head_pos[0] >= self.__grid.size or new_head_pos[1] < 0 or new_head_pos[1] >= self.__grid.size:
            self.__alive = False
            return

        tail = self.__snake[-1]
        hit = self.__grid.get_cell(new_head_pos)
        hit_any = hit is not None
        ate = hit_any and hit.value == GridCellType.APPLE
        hit_deadly = hit_any and hit is not tail and hit is not self.__apple
        
        if hit_deadly:
            # restore tail
            self.__alive = False
            return
        
        if ate:
            self._respawn_apple()
            self.__steps = 0
            
        tail = self.__snake.pop()
        old_tail_pos = tail.position
        head.value = GridCellType.SNAKE
        tail.value = GridCellType.HEAD
        tail.move(new_head_pos)
        self.__snake.appendleft(tail)
        
        self._grow_step(old_tail_pos)

        return ate

    def _grow_snake(self):
        self.__to_grow += 1

    def _grow_step(self, old_tail_pos):
        if self.__to_grow > 0:
            self.__snake.append(self.__grid.new_cell(old_tail_pos, GridCellType.SNAKE))  
            self.__to_grow -= 1  

    def _respawn_apple(self):
        self.__apple.move(self.__grid.find_free_cell())

    def has_won(self):
        return len(self.__snake) >= self.__grid.size**2
