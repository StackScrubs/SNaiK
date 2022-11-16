class Transition:
    def __init__(self, state, new_state, action, reward):
        self.state = state
        self.new_state = new_state
        self.action = action
        self.reward = reward

    def __str__(self):
        return f'state: {self.state}, new_state: {self.new_state}, action: {self.action}, reward: {self.reward}\n'
