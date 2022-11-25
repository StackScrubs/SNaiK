import torch
import torch.nn as nn
from enum import Enum

class ModelType(str, Enum):
    LINEAR = "linear"
    CONVOLUTIONAL = "convolutional"

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

    def forward(self, state):
        return self.logits(state)

    def loss(self, state, action, target_val):
        actionq_values = torch.gather(self.logits(state), 1, action.view(-1, 1))
        return nn.functional.mse_loss(actionq_values, target_val)
        return nn.functional.huber_loss(q_valurd, target_val)

    def init_layers(self):
        self.logits.apply(DQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            #nn.init.xavier_uniform_(layer.weight)
            nn.init.kaiming_uniform_(layer.weight)
            #nn.init.uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

class LinearDQN(DQN):
    def __init__(self, grid_size):
        super(LinearDQN, self).__init__()

        # self.logits = nn.Sequential(
        #     #nn.LSTM(grid_size**2*3, grid_size**2*3),
        #     nn.Flatten(),
        #     nn.Linear(grid_size**2 * 3, grid_size**2),
        #     nn.ReLU(),
        #     nn.Linear(grid_size**2, 8),

        #     nn.Softmax(1)   
        # )

        self.logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size**2 * 3, grid_size**2 * 3),
            nn.Softmax(1),
            nn.Linear(grid_size**2 * 3, grid_size**2 * 3),
            nn.Softmax(1),
            nn.Dropout1d(p=0.2),
            nn.Linear(grid_size**2 * 3, grid_size**2 * 3),
            nn.Softmax(1),
            nn.Linear(grid_size**2 * 3, 3),
            # nn.Linear(grid_size**2, 3),
            nn.Softmax(1)
        )

class ExtractTensor(nn.Module):
    def forward(self, x):
        return x[0]

class ConvolutionalDQN(DQN):
    def __init__(self, grid_size):
        super(ConvolutionalDQN, self).__init__()
        
        self.logits = nn.Sequential(
            nn.Conv2d(3, 3*24, kernel_size=(grid_size//2)+1, padding="same"),
            #nn.MaxPool2d(2),
            nn.BatchNorm2d(3*24),
            #nn.Conv2d(3*2, 3*8, kernel_size=(grid_size//2)+1, padding="same"),
            #nn.BatchNorm2d(3*8),

            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(7200, 3),
            nn.Softmax(1)
        )
