import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Przekształca (batch_size, 1, 28, 28) do (batch_size, 784)
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.model = self.model.to(torch.float64)  # Changed from float32 to float64

    def forward(self, x):
        return self.model(x)
    

import torch
import torch.nn as nn

# MLP for Wine dataset (13-16-3 architecture)
class MLP(nn.Module):
    def __init__(self, input_size=13, hidden_size=16, output_size=3):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Initialize weights in [-1, 1]
        nn.init.uniform_(self.fc1.weight, -1, 1)
        nn.init.uniform_(self.fc1.bias, -1, 1)
        nn.init.uniform_(self.fc2.weight, -1, 1)
        nn.init.uniform_(self.fc2.bias, -1, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on output
        return x

    def get_weights(self):
        # Flatten weights and biases into a vector
        return torch.cat([
            self.fc1.weight.flatten(),
            self.fc1.bias.flatten(),
            self.fc2.weight.flatten(),
            self.fc2.bias.flatten()
        ])

    def set_weights(self, weights):
        # Set weights from a flat vector
        idx = 0
        w1_size = self.input_size * self.hidden_size
        self.fc1.weight.data = weights[idx:idx+w1_size].reshape(self.hidden_size, self.input_size)
        idx += w1_size
        self.fc1.bias.data = weights[idx:idx+self.hidden_size]
        idx += self.hidden_size
        w2_size = self.hidden_size * self.output_size
        self.fc2.weight.data = weights[idx:idx+w2_size].reshape(self.output_size, self.hidden_size)
        idx += w2_size
        self.fc2.bias.data = weights[idx:idx+self.output_size]
     