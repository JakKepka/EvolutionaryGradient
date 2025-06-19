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