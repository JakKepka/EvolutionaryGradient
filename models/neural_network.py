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
     
class DeepMLP(nn.Module):
    def __init__(self, input_size=13, hidden_sizes=[64, 32, 16], output_size=3):
        super(DeepMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Tworzenie listy warstw ukrytych
        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, output_size))  # warstwa wyjściowa

        self.net = nn.Sequential(*layers)

        # Inicjalizacja wag w [-1, 1]
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1, 1)
                nn.init.uniform_(m.bias, -1, 1)

    def forward(self, x):
        return self.net(x)

    def get_weights(self):
        # Spłaszczenie wszystkich wag i biasów do jednego wektora
        params = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                params.append(m.weight.flatten())
                params.append(m.bias.flatten())
        return torch.cat(params)

    def set_weights(self, weights):
        # Ustawienie wag z wektora
        idx = 0
        for m in self.net:
            if isinstance(m, nn.Linear):
                w_shape = m.weight.shape
                b_shape = m.bias.shape
                w_num = w_shape[0] * w_shape[1]
                b_num = b_shape[0]
                m.weight.data = weights[idx:idx+w_num].reshape(w_shape)
                idx += w_num
                m.bias.data = weights[idx:idx+b_num]
                idx += b_num