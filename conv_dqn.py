import torch
import torch.nn as nn

class ExtractTensor(nn.Module):
    def forward(self, x):
        return x[0]

class ConvolutionalDQN(nn.Module):
    def __init__(self, grid_size):
        super(ConvolutionalDQN, self).__init__()
        
        self.logits = nn.Sequential(
            nn.Conv2d(3, 3*4, kernel_size=(grid_size//2)+1, padding="same"),
            nn.BatchNorm2d(3*4),
            nn.Flatten(),
            #nn.Linear(300, 300),
            #nn.Softmax(1),
            #nn.Linear(300, 300*4),
            #nn.Softmax(1),
            nn.Linear(300, 3),
            nn.Softmax(1)
        )

    def forward(self, state):
        return self.logits(state)

    def loss(self, state, action, target_val):
        actionq_values = torch.gather(self.logits(state), 1, action.view(-1, 1))
        return nn.functional.mse_loss(actionq_values, target_val)
        # return nn.functional.huber_loss(q_valurd, target_val)

    def init_layers(self):
        self.logits.apply(ConvolutionalDQN.__init_layer_weights)

    @staticmethod
    def __init_layer_weights(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            #nn.init.xavier_uniform_(layer.weight)
            nn.init.kaiming_uniform_(layer.weight)
            #nn.init.uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
