import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# Use random input (no dataset needed for gradient demo)
x = torch.randn(64, 1, 28, 28)
y = torch.randint(0, 10, (64,))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        # Bad init: large weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=5.0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

# Forward + backward (same x, y from before)
model = DeepReLU().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

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
plt.plot(range(len(layer_grads)), layer_grads, marker='o', color='red')
plt.xlabel("Layer Index (Shallow → Deep)")
plt.ylabel("Gradient Norm")
plt.title("Exploding Gradients in Deep ReLU Network (Bad Init)")
plt.grid(True)
plt.savefig("exploding_gradients.png", bbox_inches="tight")
plt.show()
