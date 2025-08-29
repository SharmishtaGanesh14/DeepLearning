# Date created: 28/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement CNN using PyTorch for image classification using cifar10 dataset
# Plot train error vs increasing number of layers

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import random

# Hyperparameters and Config

BATCH_SIZE = 64
NUM_EPOCHS = 2      # Number of epochs for each model depth
NUM_FINAL_EPOCHS = 5
LR = 0.001          # Learning rate
MOMENTUM = 0.9      # SGD momentum
CONV_KERNEL_SIZES = [5] + [3]*7  # First layer 5x5, rest 3x3
CONV_POOL_KERNEL = 2
NUM_LAYERS_START = 1
NUM_LAYERS_STOP = 8  # Inclusive: 1 to 8 conv layers
FINAL_LAYER_COUNT = 1 # Best no of layers selected from graph
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '../data'
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

# Dataset mean/std calculation
data_for_stats = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                              transform=transforms.ToTensor(), download=True)
loader_for_stats = DataLoader(data_for_stats, batch_size=5000, shuffle=False)

mean = 0
std = 0
nb_samples = 0
for data, target in loader_for_stats:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
mean = mean.numpy()
std = std.numpy()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_data = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img.cpu()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

class CNN(nn.Module):
    def __init__(self, number_of_layers=2):
        super().__init__()
        inp_channel = 3
        layers = []
        for i in range(number_of_layers):
            op_channel = 6 * (2 ** i) if i < number_of_layers - 1 else 16
            kernel_size = CONV_KERNEL_SIZES[i]
            padding = kernel_size // 2
            layers.append(nn.Conv2d(inp_channel, op_channel, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(CONV_POOL_KERNEL, CONV_POOL_KERNEL-1))
            inp_channel = op_channel

        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            conv_out = self.conv(dummy)
            self.flattened_size = conv_out.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, optimizer, loss_fn, epoch):
    model.train()
    running_loss = 0.0
    total_batches = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_batches += 1
        if i % 200 == 199:
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0
    return running_loss / total_batches

def test(model, test_loader,loss_fn):
    model.eval()
    test_loss = 0
    total_batches = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            total_batches += 1
        test_loss /= total_batches
    return test_loss

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    overall_acc = 100 * correct / total
    print(f'Overall Test Accuracy: {overall_acc:.2f}%')
    print("\nPer-class accuracy:")
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'{classname:10s}: {accuracy:.1f}%')
    return overall_acc, correct_pred, total_pred

def main():
    layer_counts = list(range(NUM_LAYERS_START, NUM_LAYERS_STOP+1))
    training_losses = []
    test_losses = []

    # Final model for visualization and accuracy
    final_model = CNN(number_of_layers=FINAL_LAYER_COUNT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(final_model.parameters(), lr=LR, momentum=MOMENTUM)
    for epoch in range(NUM_FINAL_EPOCHS):
        train_model(final_model, train_loader, optimizer, criterion, epoch)

    # Visualize some test images/labels/predictions
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    final_model.eval()
    outputs = final_model(images)
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(predicted))))

    # Overall and per-class accuracy
    evaluate_model(final_model, test_loader)

if __name__ == '__main__':
    main()
