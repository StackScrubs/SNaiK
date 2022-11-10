class Transition:
    def __init__(self, state, new_state, action, reward):
        self.state = state
        self.new_state = new_state
        self.action = action
        self.reward = reward