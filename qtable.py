import numpy as np
from pickle import dumps, loads
from transition import Transition
from typing_extensions import Self

def _discretize(grid_size: int, observation) -> int:
    n_squares = grid_size*grid_size
    dvec = lambda v: v.x*grid_size + v.y
    
    apple_obs = observation["apple"]
    apple_obs = dvec(apple_obs) if apple_obs is not None else n_squares

    return (
        dvec(observation["head"]) * n_squares**3 +
        dvec(observation["tail"]) * n_squares**2 +
        apple_obs * n_squares +
        (observation["length"] - 1)
    )

class QTable:
    def __init__(self, action_space_sz: int, state_space_sz: int) -> Self:
        self.alpha = 0.1
        self.gamma = 0.9
        self.action_space_sz = action_space_sz
        self.state_space_sz = state_space_sz
        self.__q_table = np.zeros((self.state_space_sz, self.action_space_sz))

    @classmethod
    def from_matrix(matrix: np.ndarray) -> Self:
        action_space_sz, state_space_sz = matrix.shape
        q_table = QTable(action_space_sz, state_space_sz)
        q_table.__q_table = matrix
        
        return q_table

    def update_entry(self, transition: Transition):
        self.__q_table[transition.state, transition.action] += self._bellman_equation(transition)

    def policy(self, state: int):
        return np.argmax(self.__q_table[state])

    @staticmethod
    def get_epsilon(episode: int) -> float:
        return max(0.01, min(1, 1 - np.log10((episode + 1) / 25)))

    def to_file(self, base_path = "."):
        from time import time
        pickle_dump = dumps(self)
        with open(f"{base_path}/q_model_{time()}.qbf", "wb") as f:
            f.write(pickle_dump)

    @staticmethod
    def from_file(file_path) -> Self:
        with open(file_path, "rb") as f:
            agent = loads(f.read())
            return agent

    def _bellman_equation(self, transition: Transition) -> float:
        return self.alpha*(
            transition.reward
            + self.gamma*np.max(self.__q_table[transition.new_state, :])
            - self.__q_table[transition.state, transition.action]
        )
    
class SnakeQLearningAgent:
    def __init__(self, grid_size: int):
        state_space_len = self.__get_state_space_len(grid_size)
        self.__action_space_len = 3
        self.__state = None
        self.__action = None
        self.__episode = 0
        self.__grid_size = grid_size
        self.q = QTable(self.__action_space_len, state_space_len)

    def get_optimal_action(self, observation):
        return self.q.policy(self.__discretize(observation))

    def update(self, observation, reward: float):
        new_state = self.__discretize(observation)
        self.q.update_entry(Transition(self.__state, new_state, self.__action, reward))
        action = self.__get_action(new_state)
        self.__state = new_state
        self.__action = action
        self.__episode += 1
        
        return action

    def __get_action(self, new_state):
        if np.random.random() < QTable.get_epsilon(self.__episode):
            return np.random.randint(self.__action_space_len - 1)
        else:
            return self.q.policy(new_state)
        
    def __discretize(self, observation):
        return _discretize(self.__grid_size, observation)

    @staticmethod
    def __get_state_space_len(grid_size: int) -> int:
        n_cells = grid_size**2

        n_head_pos = n_cells 
        n_apple_pos = n_cells + 1 # apple can be in any grid cells, or nowhere when game is finished
        n_tail_pos = n_cells
        n_length = n_cells
        return n_head_pos * n_apple_pos * n_tail_pos * n_length
