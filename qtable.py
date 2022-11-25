import numpy as np
from pickle import dumps, loads
from transition import Transition
from typing_extensions import Self
from discretizer import Discretizer

class QTable:
    def __init__(self, alpha: float, gamma: float, action_space_sz: int, state_space_sz: int) -> Self:
        self.alpha = alpha
        self.gamma = gamma
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
                    
    @property
    def info(self) -> dict:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
