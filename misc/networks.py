import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layers: list[int],
                 output_size: int,
                 dropout_prob: float = 0.0,
                 act_fun_str: str = "relu",
                 squash_output: bool = False):

        super(MLP, self).__init__()

        self.squash_output = squash_output
        self.layers = nn.ModuleList()

        # Check if the activation function is a string and map it to the relevant function.
        if act_fun_str == 'relu':
            self.act_fun = F.relu
        elif act_fun_str == 'tanh':
            self.act_fun = torch.tanh
        else:
            raise NotImplementedError

        # Add the first layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))

        # Add hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.Dropout(dropout_prob))

        # Add output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_fun(x)

        x = self.layers[-1](x)

        if self.squash_output:
            x = torch.tanh(x)

        return x
