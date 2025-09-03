# Date created: 28/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement CNN using PyTorch for image classification using cifar10 dataset
# Plot train error vs increasing number of layers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np

# --- Hyperparameters ---
NUM_EPOCHS = 5
NUM_FINAL_EPOCHS = 10
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
NUM_CLASSES = 10
DATA_PATH = '../data'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
NUM_LAYERS_START = 1
NUM_LAYERS_STOP = 4
VAL_SPLIT = 0.1
PATIENCE = 3

# --- CNN Model ---
class CNN(nn.Module):
    def __init__(self, number_of_layers):
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        for i in range(number_of_layers):
            out_channels = 16 * (2 ** i)
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        self.fc = nn.Linear(out_channels * 32 * 32, NUM_CLASSES)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- Compute dataset mean & std ---
data_for_stats = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                              transform=transforms.ToTensor(), download=True)
loader_for_stats = DataLoader(data_for_stats, batch_size=5000, shuffle=False)

mean = 0.0
std = 0.0
nb_samples = 0
for data, _ in loader_for_stats:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
mean /= nb_samples
std /= nb_samples
mean, std = mean.numpy(), std.numpy()

# --- Data transforms ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --- Train/Val/Test split ---
train_data_full = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                               transform=transform, download=True)
val_size = int(len(train_data_full) * VAL_SPLIT)
train_size = len(train_data_full) - val_size
train_data, val_data = random_split(train_data_full, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(RANDOM_SEED))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

test_data = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                         transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# --- Training and evaluation helpers ---
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

# --- Helper: show images + predictions ---
def imshow(img, mean, std):
    img = img.cpu()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def visualize_predictions(model, test_loader, classes, mean, std, device):
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images), mean, std)
    images, labels = images.to(device), labels.to(device)
    model.eval()
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(predicted))))

# --- Main pipeline ---
def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    # layer_counts = list(range(NUM_LAYERS_START, NUM_LAYERS_STOP + 1))
    # train_losses_final, val_losses_final = [], []
    #
    # best_val_loss, best_layer_count, best_model_state = float('inf'), None, None
    #
    # for num_layers in layer_counts:
    #     print(f"\nTraining model with {num_layers} conv layers...")
    #     model = CNN(number_of_layers=num_layers).to(DEVICE)
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    #     patience_counter = 0
    #
    #     for epoch in range(NUM_EPOCHS):
    #         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    #         val_loss, val_acc = evaluate(model, val_loader, criterion)
    #         print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
    #               f"Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}% | "
    #               f"Val Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%")
    #
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             best_layer_count = num_layers
    #             best_model_state = model.state_dict()
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= PATIENCE:
    #                 print("Early stopping triggered.")
    #                 break
    #
    #     train_losses_final.append(train_loss)
    #     val_losses_final.append(val_loss)
    #
    # # --- Plot Train vs Val Loss ---
    # plt.figure(figsize=(8, 5))
    # plt.plot(layer_counts, train_losses_final, marker='o', label='Train Loss')
    # plt.plot(layer_counts, val_losses_final, marker='s', label='Val Loss')
    # plt.title('Train/Val Loss vs Conv Layer Count')
    # plt.xlabel('Number of Conv Layers')
    # plt.ylabel('Cross-Entropy Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # --- Retrain on train+val, evaluate on test ---
    # print(f"\nBest number of layers = {best_layer_count} (Val Loss={best_val_loss:.3f})")
    full_train = ConcatDataset([train_data, val_data])
    full_train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True)

    best_layer_count=1 # found by running the above commented code

    final_model = CNN(number_of_layers=best_layer_count).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(final_model.parameters(), lr=LR, momentum=MOMENTUM)

    for epoch in range(NUM_FINAL_EPOCHS):
        train_loss, train_acc = train_one_epoch(final_model, full_train_loader, optimizer, criterion)
        print(f"[Final Training] Epoch {epoch+1}/{NUM_FINAL_EPOCHS} "
              f"- Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}%")

    test_loss, test_acc = evaluate(final_model, test_loader, criterion)
    print(f"\nFinal Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}%")

    # visualize_predictions(final_model, test_loader, classes, mean, std, DEVICE)



if __name__ == '__main__':
    main()
