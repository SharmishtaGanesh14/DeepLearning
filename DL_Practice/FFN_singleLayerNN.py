import torch
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[], activation=None):
        super(SimpleNN, self).__init__()

        # Dictionary for activations
        activation_dict = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

        layers = []
        in_size = input_size

        # Add hidden layers
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if activation is not None:
                layers.append(activation_dict[activation]())
            in_size = h  # next layer's input size

        # Output layer
        layers.append(nn.Linear(in_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Example usage
input = torch.randn(10, 3)  # 10 samples, 3 features
output = torch.randn(10, 1)  # target

# 1-layer network, no activation
model1 = SimpleNN(input_size=3, output_size=1, hidden_sizes=[], activation=None)
y_pred1 = model1(input)
print("1-layer output:\n", y_pred1)

# 2-layer network with 5 hidden units and ReLU
model2 = SimpleNN(input_size=3, output_size=1, hidden_sizes=[5], activation='relu')
y_pred2 = model2(input)
print("2-layer output:\n", y_pred2)

