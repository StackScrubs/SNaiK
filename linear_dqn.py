import torch
import torch.nn as nn
from transition import Transition
import numpy as np

class LinDQN(nn.Module):
    def __init__(self):
        super(LinDQN, self).__init__()
        self.layer_1_dim = 16
        self.layer_2_dim = 32
        self.layer_3_dim = 64

        self.logits = nn.Sequential(
            nn.Linear(4, self.layer_1_dim),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.layer_1_dim, self.layer_2_dim),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.layer_2_dim, self.layer_3_dim),
            #nn.Tanh(),
            nn.ReLU(),
            #nn.Flatten(0, -1),# ? :-()
            nn.Linear(self.layer_3_dim, 3)
        )

    def f(self, state):
        # softmax?
        #return torch.softmax(self.logits(state), dim=0)
        return self.logits(state)

    def loss(self, state, action, target_val):
        return nn.functional.mse_loss(self.logits(state)[action], target_val)
        #return (self.logits(state)[action] - target_val)**2 # based on https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function

    def init_layers(self):
        self.logits.apply(LinDQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    # def accuracy(self): ?