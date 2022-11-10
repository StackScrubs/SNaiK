import torch
import torch.nn as nn

class LinDQN(nn.Module):
    def __init__(self, state_sz):
        super(LinDQN, self).__init__()
        self.lin1 = nn.Linear(state_sz, )

    def f(self):
        pass

    def loss(self):
        pass

    #def accuracy?