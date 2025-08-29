# Date created: 29/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Download MNIST dataset and implement a MNIST classifier

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root='../data/'
seed=64
batch_size=64
epochs=5
number_of_conv_layers=1
kernel_size_first=5
kernel_size_rest=3
kernel_size_maxpool=2
stride_size_maxpool=2
fc1_out=120
fc2_out=84
fc3_out=10 # decided by number of unique layers
lr=0.001
seed=64

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        in_channels = 1

        for i in range(1, number_of_conv_layers + 1):
            op_channels = 6 + (2 ** i)  # increasing channels
            kernel_size = kernel_size_first if i == 1 else kernel_size_rest

            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=op_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size_maxpool, stride_size_maxpool))

            in_channels = op_channels

        self.conv = nn.Sequential(*layers)

        # Find flattened size by passing a dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)  # MNIST shape
            dummy_out = self.conv(dummy)
            flattened_size = dummy_out.view(1, -1).size(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flattened_size, fc1_out),
            nn.ReLU(),
            nn.Linear(fc1_out, fc2_out),
            nn.ReLU(),
            nn.Linear(fc2_out, fc3_out)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,torch.argmax(x, dim=1)


def train(model,optimiser,criterion,train_loader,epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        op,predicted = model(data)
        loss = criterion(op, target)
        loss.backward()
        optimiser.step()
        train_loss += loss.item()
        total += target.size(0)
        correct += (predicted==target).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_train_loss, accuracy

def test(model,test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(data, target) in enumerate(test_loader):
            data,target = data.to(device), target.to(device)
            op,predicted = model(data)
            test_loss += criterion(op, target).item()
            total += target.size(0)
            correct += (predicted==target).sum().item()
        test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / len(test_loader.dataset)
    print(f"Testing phase")
    print(f"Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return test_loss, accuracy

def main():
    # already ran once and found
    # dataset_for_stats = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
    # data_loader_for_stats = DataLoader(dataset_for_stats, batch_size=64, shuffle=True)
    # 
    # mean = 0.0
    # sq_mean = 0.0
    # nb_samples = 0
    # y_unique = []
    # 
    # for x, y in data_loader_for_stats:
    #     # x.shape = (batch, 1, 28, 28)
    #     batch_samples = x.size(0)
    #     x = x.view(batch_samples, -1)  # flatten
    #     mean += x.mean(1).sum(0)
    #     sq_mean += (x ** 2).mean(1).sum(0)
    #     nb_samples += batch_samples
    #     y_unique.extend(y.numpy().tolist())
    # 
    # mean /= nb_samples
    # sq_mean /= nb_samples
    # std = (sq_mean - mean ** 2).sqrt()

    # print(f"Unique labels: {len(set(y_unique))}")
    # print('Stats for data:')
    # print('mean=', mean.item())
    # print('std=', std.item())

    mean = 0.13066042959690094
    std = 0.3081077039241791

    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = torchvision.datasets.MNIST(root=root, train=True, transform=Transform)
    test_data = torchvision.datasets.MNIST(root=root, train=False, transform=Transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for epoch in range(epochs):
        avg_train_loss, accuracy = train(model,optimiser,criterion,train_loader,epoch)
    avg_test_loss, accuracy = test(model,test_loader,criterion)


if __name__ == '__main__':
    main()

            


