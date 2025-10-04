import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = None
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train(model, criterion, optimizer, loader):
    model.train()
    losses, correct, total = 0, 0, 0
    for batch, target in loader:
        batch, target = batch.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses += loss.item() * batch.size(0)
        _, pred = torch.max(output, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()
    return losses / total, correct / total

def test(model, criterion, loader):
    model.eval()
    losses, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch, target in loader:
            batch, target = batch.to(device), target.to(device)
            output = model(batch)
            loss = criterion(output, target)
            losses += loss.item() * batch.size(0)
            _, pred = torch.max(output, 1)
            total += target.size(0)
            correct += (pred == target).sum().item()
    return losses / total, correct / total

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = CNN(3, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_loss = float('inf')
patience = 10
patience_counter = 0
best_epoch = 0

for epoch in range(20):
    train_loss, train_acc = train(model, criterion, optimizer, train_loader)
    val_loss, val_acc = test(model, criterion, val_loader)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
    print(f"Val Loss: {val_loss}, Val Accuracy: {val_acc}")
    scheduler.step()
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        best_epoch = epoch + 1
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

combined_dataset = ConcatDataset([train_subset, val_subset])
combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True)

final_model = CNN(3, 10).to(device)
final_criterion = nn.CrossEntropyLoss()
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
final_scheduler = torch.optim.lr_scheduler.StepLR(final_optimizer, step_size=10, gamma=0.5)

for epoch in range(best_epoch+10):
    train_loss, train_acc = train(final_model, final_criterion, final_optimizer, combined_loader)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")
    final_scheduler.step()

test_loss, test_acc = test(final_model, final_criterion, test_loader)
print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
