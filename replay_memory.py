
class DQNAgent:
    def __init__():

        pass
    
    def _build_model():
        
        pass
    
    def memorize(): 
        
        pass
    
    def act(): #return an action
        
        pass
    
    def replay():
        
        pass

    def load():
        
        pass

    def save():
        
        pass


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)