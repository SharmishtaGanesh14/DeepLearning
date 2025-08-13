import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define model with ReLU layers
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# Load data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model and training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)

# Store activations
activations = {}

# Hook to capture outputs
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy().flatten()
    return hook

# Register hooks to capture and store the output of ReLU layers for visualization.
model.relu1.register_forward_hook(get_activation("relu1"))
model.relu2.register_forward_hook(get_activation("relu2"))

# Run one batch
model.eval()
x, y = next(iter(train_loader))
x = x.to(device)
with torch.no_grad():
    _ = model(x)

# Plot activations
plt.figure(figsize=(10, 4))
for i, name in enumerate(["relu1", "relu2"]):
    plt.subplot(1, 2, i+1)
    plt.hist(activations[name], bins=50, color='mediumseagreen', alpha=0.75)
    plt.title(f"Activation: {name}")
    plt.xlabel("Output Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.suptitle("Activation Distributions (1 Batch)", y=1.05)
plt.savefig("activations.png", bbox_inches="tight")
plt.show()
