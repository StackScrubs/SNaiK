from transition import Transition
import numpy as np
from typing_extensions import Self

class QTable:
    def __init__(self, action_space_sz: int, state_space_sz: int) -> Self:
        self.alpha = 0.001
        self.gamma = 0.99
        self.action_space_sz = action_space_sz
        self.state_space_sz = state_space_sz
        self.__q_table = np.zeros((self.state_space_sz, self.action_space_sz))

    @classmethod
    def from_matrix(matrix: np.ndarray) -> Self:
        action_space_sz, state_space_sz = matrix.shape
        q_table = QTable(action_space_sz, state_space_sz)
        q_table.__q_table = matrix
        
        return q_table

    def update_entry(self, state, transition: Transition):
        self.__q_table[state, transition.action] += self._bellman_equation(transition)

    @staticmethod
    def get_epsilon(episode: int) -> float:
        return max(.01, min(1., 1. - np.log10((episode + 1) / 25)))

    def to_file(self, base_path='.'):
        import csv
        from time import time
        with open(f'{base_path}/q_table_model_{time()}.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar=None, quoting=csv.QUOTE_NONE)
            writer.writerow([f'action_{x}' for x in range(self.action_space_sz)])

            writer.writerows(self.__q_table)

    @staticmethod
    def from_file(file_path) -> Self:
        q_table_matrix = None
        import csv
        with open(file_path, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            
            lc = 0
            for row in reader:
                if lc == 0:
                    cols = len(row)
                    q_table_matrix = np.array((0, cols))
                else:
                    np.vstack([q_table_matrix, row])
                lc += 1
            
        return QTable.from_matrix(q_table_matrix)

    def _bellman_equation(self, transition: Transition) -> float:
        return self.alpha*(
            transition.reward
            + self.gamma*np.max(self.__q_table[transition.new_state, transition.action])
            - self.__q_table[transition.state, transition.action]
        )
