import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Data loading & scaling
def load_and_clean(lm_path, tg_path):
    lm_df = pd.read_csv(lm_path, sep=",", header=None)
    tg_df = pd.read_csv(tg_path, sep=",", header=None)

    lm_df = lm_df.iloc[1:, 4:]
    tg_df = tg_df.iloc[1:, 4:]

    X = lm_df.apply(pd.to_numeric).T.values
    y = tg_df.apply(pd.to_numeric).T.values

    x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=42)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)

    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    return train_ds, val_ds, test_ds, input_dim, output_dim


# Model
class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1, hidden2, dropout):
        super(FFN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200, patience=5):
    best_val_loss = float("inf")
    counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    return best_val_loss


# Hyperparameter tuning
train_ds, val_ds, test_ds, input_dim, output_dim = load_and_clean(
    "1000G_landmark_genes.csv", "1000G_target_genes.csv"
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

param_grid = {
    "hidden1": [128, 256],
    "hidden2": [64, 128],
    "dropout": [0.2, 0.3, 0.4],
    "lr": [1e-3, 5e-4]
}

best_params = None
best_val_loss = float("inf")

criterion = nn.MSELoss()

for h1 in param_grid["hidden1"]:
    for h2 in param_grid["hidden2"]:
        for dr in param_grid["dropout"]:
            for lr in param_grid["lr"]:
                model = FFN(input_dim, output_dim, h1, h2, dr)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                val_loss = train_model(model, train_loader, val_loader, criterion, optimizer)
                print(f"Params: h1={h1}, h2={h2}, dropout={dr}, lr={lr} | Val Loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = (h1, h2, dr, lr)

print(f"\nBest Params: {best_params} | Best Val Loss: {best_val_loss:.4f}")

# Retrain on train+val, test on test
train_val_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
train_val_loader = DataLoader(train_val_ds, batch_size=32, shuffle=True)

h1, h2, dr, lr = best_params
model = FFN(input_dim, output_dim, h1, h2, dr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
train_model(model, train_val_loader, val_loader, criterion, optimizer)  # using val_loader just for early stopping

model.eval()
test_loss = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)
test_loss /= len(test_loader.dataset)

print(f"\nFinal Test Loss: {test_loss:.4f}")
