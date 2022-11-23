import torch
import torch.nn as nn
from transition import Transition
import numpy as np

class ConvolutionalDQN(nn.Module):
    def __init__(self, grid_size):
        super(ConvolutionalDQN, self).__init__()
        
        #Features: head, tail, apple, length
        self.logits = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.Flatten(),
            nn.Linear(grid_size**2*3, grid_size**2*8),
            nn.ReLU(),
            nn.Linear(grid_size**2*8, grid_size**2),
            nn.ReLU(),
            nn.Linear(grid_size**2, 3),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(32, 64, kernel_size=2),
            #nn.MaxPool2d(kernel_size=2),
            # nn.Flatten(),
            # nn.Linear(4, 1024),
            # nn.Linear(1024, 3)
        )

    def f(self, state):
        # softmax?
        eplejus = self.logits(state)
        return eplejus

    def loss(self, state, action, target_val):
        q_values = torch.gather(self.logits(state), 1, action.view(-1, 1))
        return nn.functional.mse_loss(q_values, target_val)

    def init_layers(self):
        self.logits.apply(ConvolutionalDQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            #nn.init.xavier_uniform_(layer.weight)
            #nn.init.kaiming_uniform_(layer.weight)
            nn.init.uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
