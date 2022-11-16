import torch
import torch.nn as nn
from transition import Transition

class LinDQN(nn.Module):
    def __init__(self):
        super(LinDQN, self).__init__()

        self.logits = nn.Sequential(
            nn.Linear(1, 2 * 2 * 2 * 2),
            nn.MaxPool1d(kernel_size=1),
            nn.ReLU(),
            nn.Linear(2 * 2 * 2 * 2, 4 * 4 * 4 * 4),
            nn.MaxPool1d(kernel_size=1),
            nn.ReLU(),
            # additional layer before output?
            # nn.Flatten() ? :-()
            nn.Linear(4 * 4 * 4 * 4, 3)
        )

    def f(self, state):
        # softmax?
        return self.logits(state)

    def loss(self, state, action, target_val):
        return nn.functional.mse_loss(self.logits(state)[action], target_val)
        #return (self.logits(state)[action] - target_val)**2 # based on https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function

    def init_layer_weights(self):
        self.logits.apply(LinDQN._init_layer_weights)

    def _init_layer_weights(layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    # def accuracy(self): ?