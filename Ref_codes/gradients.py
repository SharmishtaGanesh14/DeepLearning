import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Defining a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, optimizer, loss
model = SimpleNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Store gradients per layer
gradient_norms = {
    "fc1": [],
    "fc2": [],
    "fc3": []
}

# One epoch training with gradient tracking
model.train()
for batch_idx, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    # Capture gradient norms
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_name = name.split('.')[0]
            grad_norm = param.grad.norm().item()
            gradient_norms[layer_name].append(grad_norm)

    optimizer.step()

    if batch_idx > 200:  # Limiting for faster demo
        break

# Plot
plt.figure(figsize=(10, 6))
for layer, norms in gradient_norms.items():
    plt.plot(norms, label=f"{layer} weight grad norm")
plt.xlabel("Batch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norms During Training")
plt.legend()
plt.grid(True)
plt.savefig("gradients.png", bbox_inches="tight")
plt.show()
