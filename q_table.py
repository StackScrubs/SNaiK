from replay_memory import Transition
import numpy as np

class QTable:
    def __init__(self, action_space, state_space):
        self.alpha = 0.001
        self.gamma = 0.99
        self.action_space = action_space
        self.state_space = state_space
        self.q_table = np.zeros((self.state_space**2, self.action_space))

    def update_entry(self, state, action, transition: Transition):
        self.q_table[state, action] += self.bellmand_equation(transition)

    def bellmand_equation(self, transition: Transition):
        return self.alpha*(
            transition.reward
            + self.gamma*np.max(self.q_table[transition.new_state, transition.action])
            - self.q_table[transition.state, transition.action]
        )

    
        