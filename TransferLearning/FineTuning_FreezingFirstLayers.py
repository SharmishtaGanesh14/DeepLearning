from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch

# Config
BATCH_SIZE = 64
NUM_FINAL_EPOCHS = 10
LR = 0.001
MOMENTUM = 0.9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = '../data'
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
ct = 0
for child in model.children():
    ct += 1
    if ct < 5:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=LR, momentum=MOMENTUM)

# Data Loading + Train/Val Split
def load_and_split_data():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root=DATA_PATH, train=True,
                               transform=transform, download=True)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    train_val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_data = datasets.CIFAR10(root=DATA_PATH, train=False,
                                 transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, train_val_loader


# Train Step
def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(output, 1)
        correct += (preds == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# Evaluation (works for val/test)
def evaluate_model(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            for label, prediction in zip(target, preds):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, correct_pred, total_pred


# Main Training Loop
train_loader, val_loader, test_loader, train_val_loader = load_and_split_data()

for epoch in range(NUM_FINAL_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion)

    print(f"Epoch {epoch + 1}/{NUM_FINAL_EPOCHS} "
          f"- Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}% "
          f"- Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.2f}%")

# Final Test Evaluation (per-class)
test_loss, test_acc, correct_pred, total_pred = evaluate_model(model, test_loader, criterion)
print(f"\nFinal Test Accuracy: {test_acc:.2f}% (Loss: {test_loss:.3f})")

print("\nPer-class accuracy:")
for classname, correct_count in correct_pred.items():
    acc = 100 * float(correct_count) / total_pred[classname]
    print(f"{classname:5s} : {acc:.2f}%")
