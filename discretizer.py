from vec import Vec2
from math import ceil, pi

class Discretizer:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
    
    def _discretize_vec(self, vec: Vec2):
        return vec.x*self.grid_size + vec.y
    
    def state_space_len(self) -> int:
        raise NotImplementedError()
    
    @property
    def discretize(self, observation) -> int:
        raise NotImplementedError()

class FullDiscretizer(Discretizer):
    def __init__(self, grid_size: int):
        Discretizer.__init__(self, grid_size)
    
    @property
    def state_space_len(self) -> int:
        n_cells = self.grid_size**2

        n_head_pos = n_cells 
        n_apple_pos = n_cells + 1 # apple can be in any grid cells, or nowhere when game is finished
        n_tail_pos = n_cells
        n_length = n_cells
        return n_head_pos * n_apple_pos * n_tail_pos * n_length
    
    def discretize(self, observation) -> int:
        n_squares = self.grid_size*self.grid_size
        dvec = lambda v: self._discretize_vec(v)
        
        apple_obs = observation["apple"]
        apple_obs = dvec(apple_obs) if apple_obs is not None else n_squares

        return (
            dvec(observation["head"]) * n_squares**3 +
            dvec(observation["tail"]) * n_squares**2 +
            apple_obs * n_squares +
            (observation["length"] - 1)
        )
        
class AngularDiscretizer(Discretizer):
    def __init__(self, grid_size: int, n_sectors: int):
        Discretizer.__init__(self, grid_size)
        self.n_sectors = n_sectors
        self.sector_size = (2*pi)/n_sectors
        
        # Space 
        max_dist = (self.grid_size-1)*2
        self.space_wall_dist = min(max_dist, 4)
        self.space_n_dirs = 4
        self.space_n_apple_dirs = self.n_sectors + 1 # apple can be in any sector, or nowhere when game is finished
        self.space_n_tail_dirs = self.n_sectors + 1  # tail can be same position as head

    @property
    def state_space_len(self) -> int:
        max_dist = (self.grid_size-1)*2
        wall_dist_x = self.space_wall_dist
        wall_dist_y = self.space_wall_dist
        n_directions = self.space_n_dirs
        n_apple_dirs = self.space_n_apple_dirs 
        n_tail_dirs = self.space_n_tail_dirs
        return wall_dist_x * wall_dist_y * n_directions * n_apple_dirs * n_tail_dirs

    def __sectorize_angle_to_vec(self, src: Vec2, dst: Vec2) -> int:
        return int(src.angle_to(dst) // self.sector_size)
    
    def discretize(self, observation) -> int:
        # Make the state space as follows:
        # * Precise head position.
        # * Direction of snake head.
        # * Length and angle-ish to apple.
        # * Length and angle-ish to tail.
        head_pos: Vec2 = observation["head"]
        dvec = lambda v: self._discretize_vec(v)
        svec = lambda v: self.__sectorize_angle_to_vec(head_pos, v)
        
        apple_obs = observation["apple"]
        apple_obs: Vec2 = svec(observation["apple"]) if apple_obs is not None else self.n_sectors
        
        tail_obs = observation["tail"]
        tail_obs = svec(tail_obs) if head_pos != tail_obs else self.n_sectors
        
        xdistmin = min(4, min(head_pos.x < 4, head_pos.x > self.grid_size-4))
        ydistmin = min(4, min(head_pos.y < 4, head_pos.y > self.grid_size-4))
        
        return (
            xdistmin * self.space_n_dirs * self.space_n_tail_dirs * self.space_n_apple_dirs * self.space_wall_dist +
            ydistmin * self.space_n_dirs * self.space_n_tail_dirs * self.space_n_apple_dirs +
            apple_obs * self.space_n_dirs * self.space_n_tail_dirs +
            tail_obs * self.space_n_dirs +
            observation["direction"]
        )
    
class AggregatingDiscretizer(Discretizer):
    def __init__(self, grid_size: int, quad_size: int):
        Discretizer.__init__(self, grid_size)
        self.quad_size = quad_size
        self.n_axis_quads = ceil(self.grid_size / self.quad_size)
        self.n_quads = self.n_axis_quads*self.n_axis_quads
        
    @property
    def state_space_len(self) -> int:
        n_head_pos = self.grid_size**2
        n_apple_pos = self.n_quads + 1 # apple can be in any grid cells, or nowhere when game is finished
        n_tail_pos = self.n_quads
        n_directions = 4
        return n_head_pos * n_apple_pos * n_tail_pos * n_directions
    
    def __quad_discretize_vec(self, vec: Vec2) -> int:
        return (vec.x // self.quad_size) * self.n_axis_quads + vec.y // self.quad_size
    
    def discretize(self, observation) -> int:
        # Make the state space as follows:
        # * Precise head position.
        # * Quad the apple is in.
        # * Quad the tail is in.
        # * Direction of snake head.
        dvec = lambda v: self._discretize_vec(v)
        qdvec = lambda v: self.__quad_discretize_vec(v)
        
        apple_obs = observation["apple"]
        apple_obs: Vec2 = qdvec(observation["apple"]) if apple_obs is not None else self.n_quads
        
        return (
            dvec(observation["head"]) * 4 * self.n_quads**2 +
            apple_obs * 4 * self.n_quads + 
            qdvec(observation["tail"]) * 4 +
            observation["direction"]
        )
