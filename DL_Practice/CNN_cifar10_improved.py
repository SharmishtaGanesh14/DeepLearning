import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

num_classes = 10
num_layers = list(range(1, 6))
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_of_layers, num_classes=10):
        super(CNN, self).__init__()
        inp = 3   # CIFAR10 has 3 channels (RGB)
        out = 32
        layers = []
        for i in range(num_of_layers):
            layers.append(nn.Conv2d(inp, out, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2, 2))
            inp = out
            out = out * 2

        # Add adaptive pooling at the end so output is always (batch, channels, 1, 1)
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Sequential(*layers)

        # Fully connected just maps channels → num_classes
        self.fc = nn.Linear(inp, num_classes)

    def forward(self, x):
        out = self.conv(x)         # shape: (batch, channels, 1, 1)
        out = out.view(out.size(0), -1)  # flatten to (batch, channels)
        out = self.fc(out)         # final classification
        return out


def train(model, train_loader, criterion, optimizer):
    model.train()
    losses = 0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses += (loss.item() * data.size(0))
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
        total += data.size(0)
    accuracy = 100 * correct / total
    losses = losses / len(train_loader.dataset)
    return losses, accuracy


def evaluate(model, test_loader, criterion):
    model.eval()
    losses = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            losses += (loss.item() * data.size(0))
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += data.size(0)
    accuracy = 100 * correct / total
    losses = losses / len(test_loader.dataset)
    return losses, accuracy


# Normalization values (precomputed for CIFAR10)
mean = [0.4913999140262604, 0.4821586608886719, 0.4465313255786896]
std = [0.2023008167743683, 0.19941279292106628, 0.20096156001091003]

# ---------------------------
# Data Augmentation + Normalization
# ---------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

normalized_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)

val_len = int(len(normalized_dataset) * 0.2)
train_len = len(normalized_dataset) - val_len
train_dataset, val_dataset = torch.utils.data.random_split(normalized_dataset, [train_len, val_len])

train_val_loader = DataLoader(normalized_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_losses = []
val_losses = []
patience = 5

for num_layer in num_layers:
    best_loss = float('inf')
    best_accuracy = 0
    best_model_state_dict = None
    patience_counter = 0

    model = CNN(num_layer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # Learning Rate Scheduler
    # ---------------------------
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(f"[{num_layer} layers] Epoch {epoch+1}/{num_epochs} "
              f"- Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.2f}% "
              f"| Val Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.2f}%")

        scheduler.step()

        # ---------------------------
        # Save Best Model
        # ---------------------------
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_model_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping triggered")
                break

    if best_model_state_dict:
        torch.save(best_model_state_dict, f"best_model_{num_layer}layers.pth")

    train_losses.append(train_loss)
    val_losses.append(val_loss)

plt.figure()
plt.plot(num_layers, train_losses, label="Train Loss")
plt.plot(num_layers, val_losses, label="Validation Loss")
plt.xlabel("Number of Conv Layers")
plt.ylabel("Loss")
plt.title("Training and Validation Loss vs Number of Layers")
plt.grid()
plt.legend()
plt.show()

# Train final model with 5 layers
best_layer_count = 5
final_model = CNN(num_of_layers=best_layer_count).to(device)
final_optimiser = torch.optim.Adam(final_model.parameters(), lr=1e-4, weight_decay=5e-4)
final_criterion = nn.CrossEntropyLoss()
final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimiser, step_size=5, gamma=0.5)

best_state = None
best_val_acc = 0

for epoch in range(10):
    train_loss, train_acc = train(final_model, train_val_loader, final_criterion, final_optimiser)
    val_loss, val_acc = evaluate(final_model, val_loader, final_criterion)

    print(f"[Final Training] Epoch {epoch+1}/10 "
          f"- Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = final_model.state_dict()

    final_scheduler.step()

# Load best model weights before testing
if best_state:
    final_model.load_state_dict(best_state)

test_loss, test_acc = evaluate(final_model, test_loader, final_criterion)
print(f"\nFinal Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.2f}%")
