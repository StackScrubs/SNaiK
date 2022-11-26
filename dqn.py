import torch
import torch.nn as nn
from enum import Enum

class ModelType(str, Enum):
    LINEAR = "linear"
    CONVOLUTIONAL = "convolutional"

def get_next_odd_number(x: int):
    return x+(x % 2 + 1)%2

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

    def forward(self, state):
        return self.logits(state)

    def loss(self, state, action, target_val):
        actionq_values = torch.gather(self.logits(state), 1, action.view(-1, 1))
        return nn.functional.mse_loss(actionq_values, target_val)

    def init_layers(self):
        self.logits.apply(DQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.kaiming_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
            
    @property
    def info(self) -> dict:
        return {"type": self.TYPE}

class LinearDQN(DQN):
    TYPE = ModelType.LINEAR

    def __init__(self, grid_size):
        super(LinearDQN, self).__init__()

        in_features = 3 * grid_size**2
        
        self.logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, in_features * 3),
            nn.ReLU(),
            nn.Linear(in_features * 3, 3),
            nn.Softmax(1)
        )

class ConvolutionalDQN(DQN):
    TYPE = ModelType.CONVOLUTIONAL
    
    def __init__(self, grid_size):
        super(ConvolutionalDQN, self).__init__()
        
        kernel_size = get_next_odd_number(grid_size//2)
        out_channels = 3*24
        
        self.logits = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),

            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(out_channels * grid_size**2, 3),
            nn.Softmax(1)
        )
