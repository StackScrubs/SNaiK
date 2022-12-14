from __future__ import annotations
from collections import deque
from typing_extensions import Self
from typing import Any
from random import Random
from enum import IntEnum
from vec import Vec2
from itertools import islice

class Direction:
    DIRECTIONS = [
        Vec2(0, -1),  # Up
        Vec2(1, 0),  # Right
        Vec2(0, 1),  # Down
        Vec2(-1, 0),  # Left
    ]
    
    def __init__(self, direction_id):
        self.__direction_index = direction_id
        
    @property
    def vec(self) -> Vec2:
        return Direction.DIRECTIONS[self.__direction_index]
    
    @property
    def id(self) -> int:
        return self.__direction_index
        
    def turn_left(self) -> Self:
        return Direction(
            (self.__direction_index - 1 + len(self.DIRECTIONS)) % len(self.DIRECTIONS)
        )

    def turn_right(self) -> Self:
        return Direction((self.__direction_index + 1) % len(self.DIRECTIONS))   

class GridCellType(IntEnum):
    SNAKE_BODY = 1
    SNAKE_HEAD = 2
    APPLE = 3

class Grid:
    def __init__(self, size, random):
        self.size = size
        self.__cells = [None for _ in range(size*size)]
        self.__free_cells = len(self.__cells)
        self.__random = random

    def _pos_to_index(self, pos: Vec2) -> int:
        return pos.x + pos.y*self.size

    def _index_to_pos(self, idx: int) -> Vec2:
        return Vec2(idx % self.size, idx // self.size)

    def _move_cell(self, src_idx, dst_idx):
        if src_idx == dst_idx:
            return
        if self.__cells[dst_idx] is not None:
            src_pos = self._index_to_pos(src_idx)
            dst_pos = self._index_to_pos(dst_idx)
            raise Exception(
                f"Cannot move cell from {src_pos} to {dst_pos}:"
                f"Destination is occupied by {self.__cells[dst_idx]}"
            )
        self.__cells[dst_idx] = self.__cells[src_idx]
        self.__cells[src_idx] = None
        return

    def _free_cell(self, idx: int):
        self.__cells[idx] = None
        self.__free_cells += 1

    def _new_cell(self, idx: int, val: any) -> GridCell:
        if self.__cells[idx] is not None:
            raise Exception(f"Cell {self.__cells[idx]} is already occupied")
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

    def get_cell(self, pos: Vec2) -> GridCell | None:
        index = self._pos_to_index(pos)
        return self.__cells[index]

    def new_cell(self, pos: Vec2, val: any) -> GridCell:
        index = self._pos_to_index(pos)
        return self._new_cell(index, val)

    def new_free_cell(self, val: any) -> GridCell:
        return self.new_cell(self.find_free_cell(), val)

    def find_free_cell(self) -> Vec2:
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

    @property
    def value(self):
        return self.__value
    
    def move(self, pos: Vec2):
        dst_idx = self.__grid._pos_to_index(pos)
        self.__grid._move_cell(self.__index, dst_idx)
        self.__index = dst_idx

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
        self.__snake = deque([self.__grid.new_free_cell(GridCellType.SNAKE_HEAD)])
        self.__alive = True
        self.__apple = self.__grid.new_free_cell(GridCellType.APPLE)
        self.__direction = Direction(self.__random.randint(0, len(Direction.DIRECTIONS) - 1))
        self.__to_grow = 2

    @property
    def is_alive(self) -> bool:
        return self.__alive

    @property
    def head_position(self) -> Vec2:
        return self.__snake[0].position

    @property
    def tail_position(self) -> Vec2:
        return self.__snake[-1].position

    @property
    def apple_position(self) -> Vec2 | None:
        return (
            self.__apple.position
            if self.__apple is not None
            else None
        )
        
    @property
    def collidables(self) -> list[Vec2]:
        return list(b.position for b in islice(self.__snake, 4, None))

    @property
    def direction(self) -> Direction:
        return self.__direction

    @property
    def snake_length(self) -> int:
        return len(self.__snake) + self.__to_grow

    @property
    def grid_cells(self):
        return self.__grid.cells

    @property
    def has_won(self):
        return len(self.__snake) >= self.__grid.size**2

    def turn_left(self):
        self.__direction = self.__direction.turn_left()

    def turn_right(self):
        self.__direction = self.__direction.turn_right()

    def update(self):
        if not self.__alive:
            return False
        
        ate = self._snake_step()
        return ate
    
    def __kill(self):
        self.__alive = False

    def _snake_step(self) -> bool:
        head = self.__snake[0]
        
        next_head_pos = head.position + self.__direction.vec
        if not next_head_pos.within(
            Vec2(0, 0),
            Vec2(self.__grid.size - 1, self.__grid.size - 1)
        ):
            self.__kill()
            return

        hit = self.__grid.get_cell(next_head_pos)
        hit_any = hit is not None
        ate = hit_any and hit is self.__apple
        hit_deadly = hit_any and not ate

        if hit_deadly:
            self.__kill()
            return

        if ate:
            self.__apple.free()
            self.__apple = None
            self.__to_grow += 1

        old_head = head
        new_head = None
        if self.__to_grow == 0:
            new_head = self.__snake.pop()
            new_head.move(next_head_pos)
        else:
            new_head = self.__grid.new_cell(next_head_pos, GridCellType.SNAKE_HEAD)
            self.__to_grow -= 1

        old_head.value = GridCellType.SNAKE_BODY
        new_head.value = GridCellType.SNAKE_HEAD

        self.__snake.appendleft(new_head)

        if ate and not self.has_won:
            self._spawn_apple()

        return ate

    def _grow_snake(self):
        self.__to_grow += 1

    def _spawn_apple(self):
        self.__apple = self.__grid.new_free_cell(GridCellType.APPLE)
