import torch
import torch.nn as nn
from transition import Transition
import numpy as np

class ConvolutionalDQN(nn.Module):
    def __init__(self, grid_size):
        super(ConvolutionalDQN, self).__init__()
        
        self.logits = nn.Sequential(
            # Scrubs: Es funktioniert! :)
            # Snake : Bitte
            nn.Flatten(),
            nn.Linear(grid_size**2*3, 3),
            nn.Softmax(1)
        )

    def forward(self, state):
        eplejus = self.logits(state)
        return eplejus

    def loss(self, state, action, target_val):
        actionq_values = torch.gather(self.logits(state), 1, action.view(-1, 1))
        return nn.functional.mse_loss(actionq_values, target_val)
        return nn.functional.huber_loss(q_valurd, target_val)

    def init_layers(self):
        self.logits.apply(ConvolutionalDQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            #nn.init.xavier_uniform_(layer.weight)
            #nn.init.kaiming_uniform_(layer.weight)
            nn.init.uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
