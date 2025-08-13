import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the same SimpleNN
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Load data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model for a few epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(3):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

# Plot histograms of the weights
plt.figure(figsize=(12, 4))
for i, name in enumerate(['fc1', 'fc2', 'fc3']):
    weights = getattr(model, name).weight.data.cpu().numpy().flatten()
    plt.subplot(1, 3, i+1)
    plt.hist(weights, bins=50, alpha=0.75, color='royalblue')
    plt.title(f"Weights of {name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.suptitle("Weight Distributions After Training", y=1.05)
plt.savefig("inspecting_weights.png", bbox_inches="tight")
plt.show()
