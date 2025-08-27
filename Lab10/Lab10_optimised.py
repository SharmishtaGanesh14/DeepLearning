import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import Adam
from torch import nn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Normalise_X:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)

    def __call__(self, x):
        return (x - self.mean) / self.std

class Normalise_Y:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float)
        self.std = torch.tensor(std, dtype=torch.float)

    def __call__(self, y):
        return (y - self.mean) / self.std

class GeneExpressionDataset(Dataset):
    def __init__(self, x, y, transform_x=None, transform_y=None):
        self.X = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        return x, y

def preprocessing(path1, path2):
    seed = 42
    lm_df = pd.read_csv(path1, header=0).iloc[:, 4:]
    tg_df = pd.read_csv(path2, header=0).iloc[:, 4:]

    X = lm_df.values.T.astype(np.float32)  # Convert to float32 numpy array
    y = tg_df.values.T.astype(np.float32)

    x_train, x_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=seed)
    x_test, x_val, y_test, y_val = train_test_split(x_rest, y_rest, test_size=0.5, random_state=seed)

    mean_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0) + 1e-8  # Avoid division by zero

    mean_y = np.mean(y_train, axis=0)
    std_y = np.std(y_train, axis=0) + 1e-8

    norm_x = Normalise_X(mean_x, std_x)
    norm_y = Normalise_Y(mean_y, std_y)

    x_train_ds = GeneExpressionDataset(x_train, y_train, transform_x=norm_x, transform_y=norm_y)
    x_val_ds = GeneExpressionDataset(x_val, y_val, transform_x=norm_x, transform_y=norm_y)
    x_test_ds = GeneExpressionDataset(x_test, y_test, transform_x=norm_x, transform_y=norm_y)

    return x_train_ds, x_val_ds, x_test_ds, X.shape[1], y.shape[1]


class FFN(nn.Module):
    def __init__(self, in_features, out_features, h1, h2, dr):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.Dropout(dr),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.Dropout(dr),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Linear(h2, out_features)
        )

    def forward(self, x):
        return self.network(x)


def train(data_loader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(data_loader.dataset)


def evaluate(data_loader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_fn(pred, target)
            total_loss += loss.item() * data.size(0)
    return total_loss / len(data_loader.dataset)


def main():
    train_data, val_data, test_data, input_dim, output_dim = preprocessing(
        path1="1000G_landmark_genes.csv", path2="1000G_target_genes.csv")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    loss_fn = nn.MSELoss()

    space = {
        "hidden1": hp.choice("hidden1", [128, 256, 512]),
        "hidden2": hp.choice("hidden2", [64, 128, 256]),
        "dropout": hp.uniform("dropout", 0.1, 0.5),
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-2))  # 1e-5 to 1e-2 typical range
    }

    def objective(params):
        model = FFN(in_features=input_dim, out_features=output_dim,
                    h1=params["hidden1"], h2=params["hidden2"], dr=params["dropout"]).to(device)
        optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)
        for epoch in range(30):
            train(train_loader, model, loss_fn, optimizer)
        val_loss = evaluate(val_loader, model, loss_fn)
        return {"loss": val_loss, "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials)

    h1_candidates = [128, 256, 512]
    h2_candidates = [64, 128, 256]

    h1 = h1_candidates[best["hidden1"]]
    h2 = h2_candidates[best["hidden2"]]
    dr = best["dropout"]
    lr = best["lr"]

    print(f"\nUsing: hidden1={h1}, hidden2={h2}, dropout={dr:.3f}, lr={lr:.6f}")

    final_model = FFN(input_dim, output_dim, h1, h2, dr).to(device)
    final_optimizer = Adam(final_model.parameters(), lr=lr, weight_decay=1e-5)

    train_val_loader = DataLoader(ConcatDataset([train_data, val_data]), batch_size=32, shuffle=True)
    train_losses = []

    epochs = 300
    for t in range(epochs):
        train_loss = train(train_val_loader, final_model, loss_fn, final_optimizer)
        train_losses.append(train_loss)
        if (t + 1) % 50 == 0:
            print(f"Epoch {t + 1} | Train Loss: {train_loss:.4f}")

    test_loss = evaluate(test_loader, final_model, loss_fn)
    print(f"\nFinal Test Loss: {test_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    torch.save(final_model.state_dict(), "tg_prediction_from_lm_genes.pth")
    print("Saved final model to tg_prediction_from_lm_genes.pth")


if __name__ == "__main__":
    main()
