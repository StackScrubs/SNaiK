import torch
import torch.nn as nn
from transition import Transition
import numpy as np

class ConvolutionalDQN(nn.Module):
    def __init__(self):
        super(ConvolutionalDQN, self).__init__()
        
        #Features: head, tail, apple, length
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(32, 64, kernel_size=2),
            #nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32, 1024),
            nn.Linear(1024, 3)
        )

    def f(self, state):
        # softmax?
        #return torch.softmax(self.logits(state), dim=0)
        #state.to(torch.device("cpu"))
        #return self.logits(state)
        eplejus = self.logits(state)
        #print(eplejus)
        return eplejus

    def loss(self, state, action, target_val):
        #print(self.logits(state))
        #print(self.logits(state)[0][action])
        #return nn.functional.cross_entropy(self.logits(state), target_val)
        return nn.functional.mse_loss(self.logits(state)[0][action], target_val)
        #return (self.logits(state)[action] - target_val)**2 # based on https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function

    def init_layers(self):
        self.logits.apply(ConvolutionalDQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    # def accuracy(self): ?