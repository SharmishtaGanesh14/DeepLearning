# Date created: 13/08/2025
# Date updated: 15/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan

# Aim: Develop a model to predict target genes from a set of landmark genes
# Explore different ways of encoding input data.
# Tune hyperparameters for improved performance.

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading
def load_and_clean(lm_path, tg_path,n_pca_components=None):
    lm_df = pd.read_csv(lm_path, header=None).iloc[1:, 4:]
    tg_df = pd.read_csv(tg_path, header=None).iloc[1:, 4:]

    X = lm_df.apply(pd.to_numeric).T.values
    y = tg_df.apply(pd.to_numeric).T.values

    x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.66, random_state=42)
    x_val2,x_final_test,y_val2,y_final_test = train_test_split(x_val, y_val, test_size=0.5, random_state=42)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_val2 = x_scaler.transform(x_val2)
    x_final_test = x_scaler.transform(x_final_test)
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_val2 = y_scaler.transform(y_val2)
    y_final_test = y_scaler.transform(y_final_test)

    if n_pca_components is not None:
        pca = PCA(n_components=n_pca_components)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_val2 = pca.transform(x_val2)
        x_final_test = pca.transform(x_final_test)
        input_dim = n_pca_components  # Changed input dimension after PCA
    else:
        input_dim = X.shape[1]
    # PCA has no effect

    def to_ds(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.float32))

    return to_ds(x_train, y_train), to_ds(x_val, y_val), to_ds(x_val2, y_val2),to_ds(x_final_test,y_final_test),input_dim, y.shape[1]

# Model
class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=256, hidden2=128, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# Training & Testing
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

def test(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

def main():
    train_ds, val_ds, val2_ds,test_ds, input_dim, output_dim = load_and_clean(
        "1000G_landmark_genes.csv", "1000G_target_genes.csv")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    val2_loader = DataLoader(val2_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)

    param_grid = {
        "hidden1": [128, 256],
        "hidden2": [64, 128],
        "dropout": [0.2, 0.3],
        "lr": [1e-3, 5e-4]
    }

    loss_fn = nn.MSELoss()

    best_params, best_val_loss = None, float("inf")
    for h1 in param_grid["hidden1"]:
        for h2 in param_grid["hidden2"]:
            for dr in param_grid["dropout"]:
                for lr in param_grid["lr"]:
                    model = FFN(input_dim, output_dim, h1, h2, dr).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                    # Short training for tuning
                    for epoch in range(50):
                        train(train_loader, model, loss_fn, optimizer)
                    val_loss = test(val_loader, model, loss_fn)
                    print(f"h1={h1}, h2={h2}, dr={dr}, lr={lr} | Val Loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = (h1, h2, dr, lr)

    print(f"\nBest Params: {best_params} | Best Val Loss: {best_val_loss:.4f}")

    # Final training
    train_val_loader = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=32, shuffle=True)

    # 256, 128, 0.2, 0.001
    h1, h2, dr, lr = best_params
    model = FFN(input_dim, output_dim, h1, h2, dr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    epochs = 300
    train_losses,test_losses = [], []

    for t in range(epochs):
        train_loss = train(train_val_loader, model, loss_fn, optimizer)
        test_loss=test(val2_loader, model, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if (t+1) % 50 == 0:
            print(f"Epoch {t+1} | Train Loss: {train_loss:.4f}) | Test Loss: {test_loss:.4f}")

    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "tg_prediction_from_lm_genes.pth")
    print("Saved PyTorch Model State to tg_prediction_from_lm_genes.pth")

    # Final Test Evaluation
    # Load model from saved file
    final_model = FFN(input_dim, output_dim, h1, h2, dr).to(device)
    final_model.load_state_dict(torch.load("tg_prediction_from_lm_genes.pth"))
    final_model.eval()

    # Evaluate on final test set
    final_test_loss = test(test_loader, final_model, loss_fn)
    print(f"\nFinal Test Loss (unseen data): {final_test_loss:.4f}")

if __name__ == "__main__":
    main()
