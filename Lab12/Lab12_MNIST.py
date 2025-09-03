# Date created: 29/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Download MNIST dataset and implement a MNIST classifier

import random

import numpy as np
import torch
import torchvision
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def set_seed(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Config
root = '../data/'
seed = 64
batch_size = 64
epochs = 5
number_of_conv_layers = 1
kernel_size_first = 5
kernel_size_rest = 3
kernel_size_maxpool = 2
stride_size_maxpool = 2
fc3_out = 10  # number of unique classes

# Apply seed
set_seed(seed)


class CNN(nn.Module):
    def __init__(self, fc1_out, fc2_out):
        super().__init__()
        layers = []
        in_channels = 1

        for i in range(1, number_of_conv_layers + 1):
            op_channels = 6 + (2 ** i)
            kernel_size = kernel_size_first if i == 1 else kernel_size_rest
            layers.append(nn.Conv2d(in_channels, op_channels,
                                    kernel_size=kernel_size,
                                    stride=1, padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size_maxpool, stride_size_maxpool))
            in_channels = op_channels

        self.conv = nn.Sequential(*layers)

        # Flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            dummy_out = self.conv(dummy)
            flattened_size = dummy_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, fc1_out),
            nn.BatchNorm1d(fc1_out),
            nn.ReLU(),
            nn.Linear(fc1_out, fc2_out),
            nn.BatchNorm1d(fc2_out),
            nn.ReLU(),
            nn.Linear(fc2_out, fc3_out)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # logits only


def train(model, optimiser, criterion, train_loader, epoch):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total, correct, test_loss = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            test_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    avg_loss = test_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


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

    mean, std = 0.13066042959690094, 0.3081077039241791
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load dataset
    full_train = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST(root=root, train=False, transform=transform, download=True)

    # Split train/val
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    training, val = random_split(full_train, [train_size, val_size])

    train_val_loader = DataLoader(full_train, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Hyperopt space
    space = {
        "hidden1": hp.choice("hidden1", [128, 256, 512]),
        "hidden2": hp.choice("hidden2", [64, 128, 256]),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2))
    }

    criterion = nn.CrossEntropyLoss()

    def objective(params):
        model = CNN(params["hidden1"], params["hidden2"]).to(device)
        optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
        for epoch in range(epochs):
            train(model, optimizer, criterion, train_loader, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        return {"loss": val_loss, "status": STATUS_OK, "accuracy": val_acc}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials)

    h1_candidates = [128, 256, 512]
    h2_candidates = [64, 128, 256]

    h1 = h1_candidates[best["hidden1"]]
    h2 = h2_candidates[best["hidden2"]]
    lr = best["lr"]

    print(f"\nBest Hyperparams -> hidden1={h1}, hidden2={h2}, lr={lr:.6f}")

    # Final model
    model_final = CNN(h1, h2).to(device)
    optimiser_final = Adam(model_final.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        train(model_final, optimiser_final, criterion, train_val_loader, epoch)

    test_loss, test_acc = evaluate(model_final, test_loader, criterion)
    print(f"\nTesting phase -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()