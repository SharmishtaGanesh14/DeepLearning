# Imports
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Reproducibility
# -----------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Data preprocessing
def get_dataloaders(X, y, batch_size=64):
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=seed
    )
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=seed
    )

    # Create datasets
    train_dataset = ClassificationDataset(X_train, y_train)
    val_dataset = ClassificationDataset(X_val, y_val)
    test_dataset = ClassificationDataset(X_test, y_test)
    train_val_dataset = ClassificationDataset(X_train_val, y_train_val)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, train_val_loader

# Feedforward Neural Network
class FFN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512,256,128], 
                 activation='relu', dropout=0.5):
        super().__init__()

        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation {activation} not supported")

        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, output_size))  # Output layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Training and testing functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Main
# Generate synthetic classification data
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=15, n_redundant=5, n_classes=2, random_state=seed
)

train_loader, val_loader, test_loader, train_val_loader = get_dataloaders(X, y)

model = FFN(input_size=X.shape[1], output_size=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early stopping parameters
patience = 10
best_val_loss = np.inf
patience_counter = 0
train_losses, val_losses = [], []

# Training with early stopping
max_epochs = 50
for epoch in range(max_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save best model
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# Plot training curves
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Final training on full train+val set
final_model = FFN(input_size=X.shape[1], output_size=2).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
final_criterion = nn.CrossEntropyLoss()

final_epochs = len(train_losses)  # same number of epochs as before
for epoch in range(final_epochs):
    train_loss, train_acc = train_one_epoch(final_model, train_val_loader, final_optimizer, final_criterion)
    print(f"[Final Training] Epoch {epoch+1}/{final_epochs} | Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")

# Evaluate on test set
test_loss, test_acc = evaluate(final_model, test_loader, final_criterion)
print(f"\n[Test Set] Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
