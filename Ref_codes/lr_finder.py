import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define a Simple Neural Network
# Simple feedforward neural network with two hidden layers
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Learning Rate Finder Logic
# Try different learning rates and track the loss
def find_lr(model, train_loader, init_lr=1e-5, final_lr=10, beta=0.98):
    num = len(train_loader) - 1
    mult = (final_lr / init_lr) ** (1/num)
    lr = init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    avg_loss = 0.
    best_loss = float('inf')
    batch_num = 0
    losses = []
    log_lrs = []

    for inputs, labels in train_loader:
        batch_num += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Stop if the loss explodes
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # Record the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))

        # Backprop
        loss.backward()
        optimizer.step()

        # Update the LR
        lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return log_lrs, losses


# Run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN().to(device)
log_lrs, losses = find_lr(model, train_loader)

# Plot
plt.plot(log_lrs, losses)
plt.xlabel("Log Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate Finder")
plt.grid(True)
plt.savefig("lr_finder.png", bbox_inches="tight")
plt.show()

