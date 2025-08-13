import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Deep Sigmoid MLP (6 layers)
class DeepSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

# Use random input (no dataset needed for gradient demo)
x = torch.randn(64, 1, 28, 28)
y = torch.randint(0, 10, (64,))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepSigmoid().to(device)
x, y = x.to(device), y.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Forward + backward
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()

# Collect gradient norms
layer_grads = []
for name, param in model.named_parameters():
    if 'weight' in name:
        layer_grads.append(param.grad.norm().item())

# Plot
plt.plot(range(len(layer_grads)), layer_grads, marker='o')
plt.xlabel("Layer Index (Shallow → Deep)")
plt.ylabel("Gradient Norm")
plt.title("Vanishing Gradients in Deep Sigmoid Network")
plt.grid(True)
plt.savefig("vanishing_gradients.png", bbox_inches="tight")
plt.show()
