from torch import nn
import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# Device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# Dataset wrapper
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # features
        self.y = torch.tensor(y, dtype=torch.long)  # labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Feedforward Neural Network
class FFN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation='relu', dropout=0.2):
        super().__init__()
        if activation == 'relu':
            act_fn = nn.ReLU()  # choose activation
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [input_size] + hidden_sizes
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))  # linear layer
            seq.append(nn.BatchNorm1d(layers[i + 1]))  # batch norm
            seq.append(act_fn)  # activation
            seq.append(nn.Dropout(dropout))  # dropout
        seq.append(nn.Linear(layers[-1], output_size))  # final output
        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)  # forward pass


# Train one epoch
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    losses, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)  # compute loss
        loss.backward()
        optimizer.step()  # update weights
        losses += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return losses / len(loader.dataset), 100 * correct / total


# Evaluate model
def eval_model(model, loader, criterion):
    model.eval()
    losses, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            losses += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return losses / len(loader.dataset), 100 * correct / total


# Prepare data
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=2, random_state=seed)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=seed)

# Create loaders
train_loader = DataLoader(ClassificationDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(ClassificationDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(ClassificationDataset(X_test, y_test), batch_size=64, shuffle=False)
train_val_loader = DataLoader(ClassificationDataset(X_train_val, y_train_val), batch_size=64, shuffle=True)


# Hyperopt objective
def objective(params):
    hidden_sizes = [int(params['hidden1']), int(params['hidden2'])]
    dropout = params['dropout']
    lr = params['lr']

    model = FFN(X_train.shape[1], 2, hidden_sizes, 'relu', dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    best_val_loss = float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0


    # Train few epochs
    for epoch in range(20):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_model(model, val_loader, criterion)
        train_losses.append(tr_loss)  # store train loss
        val_losses.append(val_loss)  # store val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        'loss': -val_acc,  # maximize accuracy
        'status': STATUS_OK,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_acc': val_acc,
        'best_epoch': best_epoch
    }


# Hyperparameter space
space = {
    'hidden1': hp.quniform('hidden1', 64, 512, 32),
    'hidden2': hp.quniform('hidden2', 64, 512, 32),
    'dropout': hp.uniform('dropout', 0.2, 0.6),
    'lr': hp.loguniform('lr', np.log(1e-4), np.log(1e-2))
}

# Run hyperopt
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials,
            rstate=np.random.default_rng(seed))
print("Best Hyperparameters:", best)

# Plot train/val loss for best trial
best_idx = np.argmin([t['result']['loss'] for t in trials.trials])
best_trial = trials.trials[best_idx]['result']
best_epoch = best_trial['best_epoch']
plt.plot(best_trial['train_losses'], label='Train Loss')
plt.plot(best_trial['val_losses'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.legend()
plt.show()

# Final training on full train+val set
hidden_sizes = [int(best['hidden1']), int(best['hidden2'])]
dropout = best['dropout']
lr = best['lr']

model = FFN(X_train.shape[1], 2, hidden_sizes, 'relu', dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
for epoch in range(best_epoch+1):
    tr_loss, tr_acc = train_epoch(model, train_val_loader, optimizer, criterion)
    train_losses.append(tr_loss)
    print(f"[Final Train] Epoch {epoch + 1}: Loss={tr_loss:.4f}, Acc={tr_acc:.2f}")

# Final evaluation on test set
test_loss, test_acc = eval_model(model, test_loader, criterion)
print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}")
