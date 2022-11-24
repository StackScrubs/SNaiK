import numpy as np
from pickle import dumps, loads
from transition import Transition
from typing_extensions import Self
from discretizer import Discretizer

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

    def _bellman_equation(self, transition: Transition) -> float:
        return self.alpha*(
            transition.reward
            + self.gamma*np.max(self.__q_table[transition.new_state, :])
            - self.__q_table[transition.state, transition.action]
        )
        
    def __str__(self) -> str:
        return f"A={self.alpha}_G={self.gamma}"
    
class SnakeQLearningAgent:
    def __init__(self, discretizer: Discretizer):
        self.__action_space_len = 3
        self.__state = None
        self.__action = None
        self.__episode = 0
        self.discretizer = discretizer
        self.q = QTable(self.__action_space_len, discretizer.state_space_len)

    def get_optimal_action(self, observation):
        return self.q.policy(self.discretizer.discretize(observation))

    def update(self, observation, reward: float):
        new_state = self.discretizer.discretize(observation)
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
    
    def to_file(self, base_path = ".") -> str:
        from time import time
        pickle_dump = dumps(self)
        file_name = f"{base_path}/q_model_{time()}.qbf"
        with open(file_name, "wb") as f:
            f.write(pickle_dump)

        return file_name

    @staticmethod
    def from_file(file_path) -> Self:
        with open(file_path, "rb") as f:
            agent = loads(f.read())
            return agent
        
    def __str__(self) -> str:
        return str(self.q) + f"_SZ={self.discretizer.grid_size}_QLEARNING" + str(self.discretizer)
        
