from vec import Vec2
from math import ceil, pi
from enum import Enum
from snake_state import Direction

class DiscretizerType(str, Enum):
    FULL = "full"
    QUAD = "quad"
    ANGULAR = "angular"

class Discretizer:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
    
    def _discretize_vec(self, vec: Vec2):
        return vec.x*self.grid_size + vec.y
                
    @property
    def info(self) -> dict:
        return {"type": self.TYPE}


"""
Discretizes the entire grid into a discrete state. This contains the full state of the grid.
""" 
class FullDiscretizer(Discretizer):
    TYPE = DiscretizerType.FULL
    
    def __init__(self, grid_size: int):
        Discretizer.__init__(self, grid_size)
    
    @property
    def state_space_len(self) -> int:
        n_apple_pos = self.grid_size**2 + 1
        n_head_pos = self.grid_size**2 + 1
        n_cell_combinations = 2**(self.grid_size**2)
        
        return n_apple_pos * n_head_pos * n_cell_combinations
    
    def discretize(self, observation) -> int:             
        dvec = lambda v: self._discretize_vec(v)  
        head = observation["head"]
        apple = observation["apple"] 
        grid = observation["grid"]
        
        apple_max = self.grid_size**2 if apple is not None else (self.grid_size**2)+1
        apple_pos = dvec(apple) if apple is not None else self.grid_size**2
        head_pos = dvec(head)
        
        v = apple_pos + head_pos * apple_max
        for cell in grid.flat:
            if cell not in (0, 1):
                continue
            v = v << 1 | cell
        return v


""" 
QuadDiscretizer discretizes state into a certain amount of quadrants, where the snake is aware of which 
quadrant the apple and tail is located in, which way the snake is headed, as well as the exact location of the head.  
"""
class QuadDiscretizer(Discretizer):
    TYPE = DiscretizerType.QUAD
    
    def __init__(self, grid_size: int, quad_size: int):
        Discretizer.__init__(self, grid_size)
        self.quad_size = quad_size
        self.n_axis_quads = ceil(self.grid_size / self.quad_size)
        self.n_quads = self.n_axis_quads**2
        
    @property
    def state_space_len(self) -> int:
        n_head_pos = self.grid_size**2
        n_apple_pos = self.n_quads + 1 # apple can be in any quads, or nowhere when game is finished
        n_tail_pos = self.n_quads
        n_directions = 4
        return n_head_pos * n_apple_pos * n_tail_pos * n_directions
    
    def __quad_discretize_vec(self, vec: Vec2) -> int:
        return (vec.x // self.quad_size) * self.n_axis_quads + (vec.y // self.quad_size)
    
    def discretize(self, observation) -> int:
        dvec = lambda v: self._discretize_vec(v)
        qdvec = lambda v: self.__quad_discretize_vec(v)
        
        apple_obs = observation["apple"]
        apple_obs: Vec2 = qdvec(observation["apple"]) if apple_obs is not None else self.n_quads
        
        return (
            dvec(observation["head"]) * 4 * self.n_quads**2 +
            apple_obs * 4 * self.n_quads + 
            qdvec(observation["tail"]) * 4 +
            observation["direction"].id
        )
                    
    @property
    def info(self) -> dict:
        return {
            **super().info,
            "quad_size": self.quad_size
        }

"""
AngularDiscretizer discretizes state as relative angle, encoded into circle sectors, and position of objects as related to snake's head.
"""
class AngularDiscretizer(Discretizer):
    TYPE = DiscretizerType.ANGULAR
    
    def __init__(self, grid_size: int, n_sectors: int):
        Discretizer.__init__(self, grid_size)
        self.n_sectors = n_sectors
        self.sector_size = (2*pi)/n_sectors
        
        max_dist = (self.grid_size-1)*2
        self.clamped_wall_dists = min(max_dist, 4)
        self.clamped_tail_dists = min(max_dist, 4) + 1 # collidable tail can be non-existent
        self.n_dirs = 4
        self.n_apple_dirs = self.n_sectors + 1 # apple can be in any sector, or nowhere when game is finished
        self.n_tail_dirs = self.n_sectors + 1  # there can be no tails that the snake can collide with
        
    @property
    def state_space_len(self) -> int:
        wall_dist_x = self.clamped_wall_dists
        wall_dist_y = self.clamped_wall_dists
        tail_dist = self.clamped_tail_dists
        n_directions = self.n_dirs
        n_apple_dirs = self.n_apple_dirs
        n_tail_dirs = self.n_tail_dirs
        danger_flag = 2**4
        return wall_dist_x * wall_dist_y * n_directions * n_apple_dirs * n_tail_dirs * tail_dist * danger_flag

    def __sectorize_angle_between_vecs(self, src: Vec2, dst: Vec2) -> int:
        return int(src.angle_to(dst) // self.sector_size)
    
    def discretize(self, observation) -> int:
        direction: Direction = observation["direction"]
        head_pos: Vec2 = observation["head"]
        svec = lambda v: self.__sectorize_angle_between_vecs(head_pos, v)
        
        apple_obs = observation["apple"]
        apple_obs: Vec2 = svec(observation["apple"]) if apple_obs is not None else self.n_sectors
        
        dangers = {
            "front": [head_pos + direction.vec, 0],
            "left": [head_pos + direction.turn_left().vec, 0],
            "right": [head_pos + direction.turn_right().vec, 0]
        }
        
        closest_tail_dist = float("inf")
        closest_tail = None
        for c in observation["collidables"]:
            dist = head_pos.manhattan_dist(c)
            if dist <= closest_tail_dist:
                closest_tail_dist = dist
                closest_tail = c
            for danger_name in dangers:
                danger = dangers[danger_name]
                if c == danger[0]:
                    danger[1] = 1
        
        tail_obs = svec(closest_tail) if closest_tail is not None else self.n_tail_dirs - 1
        tail_dist = closest_tail_dist if closest_tail is not None else self.clamped_tail_dists - 1

        danger_flags = dangers["front"][1] << 2 | dangers["left"][1] << 1 | dangers["right"][1]

        wall_dist_min_x = min(4, min(head_pos.x < 4, head_pos.x > self.grid_size-4))
        wall_dist_min_y = min(4, min(head_pos.y < 4, head_pos.y > self.grid_size-4))
        
        return (
            danger_flags     * self.n_dirs * self.n_tail_dirs * self.n_apple_dirs * self.clamped_wall_dists * self.clamped_wall_dists * self.clamped_tail_dists +
            tail_dist        * self.n_dirs * self.n_tail_dirs * self.n_apple_dirs * self.clamped_wall_dists * self.clamped_wall_dists +
            wall_dist_min_x  * self.n_dirs * self.n_tail_dirs * self.n_apple_dirs * self.clamped_wall_dists +
            wall_dist_min_y  * self.n_dirs * self.n_tail_dirs * self.n_apple_dirs +
            apple_obs        * self.n_dirs * self.n_tail_dirs +
            tail_obs         * self.n_dirs +
            observation["direction"].id
        )
                    
    @property
    def info(self) -> dict:
        return {
            **super().info,
            "n_sectors": self.n_sectors,
        }
