import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Data
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1,random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
torch.manual_seed(42)

# Deep MLP Model
class DeepMLP(nn.Module):
    def __init__(self, activation='sigmoid', n_hidden=10, n_layers=10):
        super().__init__()
        activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}
        layers = [nn.Linear(2, n_hidden), activations[activation]]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activations[activation])
        layers.append(nn.Linear(n_hidden, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Training with capture
def train_and_capture(model, X, y, n_epochs=50, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    weight_mean_history, weight_std_history = [], []
    grad_mean_history, grad_std_history = [], []
    loss_history = []
    acc_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        acc = ((y_pred > 0.5).float() == y).float().mean().item()

        # Capture weight and gradient stats per layer
        w_mean, w_std = [], []
        g_mean, g_std = [], []
        for name, param in model.named_parameters():
            if 'weight' in name:
                w = param.detach().numpy()
                g = param.grad.detach().numpy()
                w_mean.append(w.mean())
                w_std.append(w.std())
                g_mean.append(g.mean())
                g_std.append(g.std())

        weight_mean_history.append(w_mean)
        weight_std_history.append(w_std)
        grad_mean_history.append(g_mean)
        grad_std_history.append(g_std)
        loss_history.append(loss.item())
        acc_history.append(acc)

    return weight_mean_history, weight_std_history, grad_mean_history, grad_std_history, loss_history,acc_history


# Plot all stats in one figure
def plot_all_stats(weight_mean, weight_std, grad_mean, grad_std, loss, activation):
    epochs = range(len(loss))
    weight_mean = list(zip(*weight_mean))
    weight_std = list(zip(*weight_std))
    grad_mean = list(zip(*grad_mean))
    grad_std = list(zip(*grad_std))

    # 2x2 figure for weights and gradients
    plt.figure(figsize=(14, 12))

    # Row 1, Col 1: Weight Mean
    plt.subplot(2, 2, 1)
    for i, wm in enumerate(weight_mean):
        plt.plot(epochs, wm, label=f'Layer {i + 1}')
    plt.title(f'Weight Mean per Layer ({activation})')
    plt.legend()

    # Row 1, Col 2: Weight Std
    plt.subplot(2, 2, 2)
    for i, ws in enumerate(weight_std):
        plt.plot(epochs, ws, label=f'Layer {i + 1}')
    plt.title(f'Weight Std per Layer ({activation})')
    plt.legend()

    # Row 2, Col 1: Gradient Mean
    plt.subplot(2, 2, 3)
    for i, gm in enumerate(grad_mean):
        plt.plot(epochs, gm, label=f'Layer {i + 1}')
    plt.title(f'Gradient Mean per Layer ({activation})')
    plt.legend()

    # Row 2, Col 2: Gradient Std (log scale)
    plt.subplot(2, 2, 4)
    for i, gs in enumerate(grad_std):
        plt.semilogy(epochs, gs, label=f'Layer {i + 1}')
    plt.title(f'Gradient Std per Layer ({activation})')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Separate figure for Loss
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, loss, color='black')
    plt.title(f'Loss ({activation})')
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.tight_layout()
    plt.show()

# Run for different activations
for act in ['sigmoid', 'tanh', 'relu']:
    print(f"Activation: {act}")
    model = DeepMLP(activation=act, n_layers=5)
    w_mean, w_std, g_mean, g_std, loss,acc = train_and_capture(model, X, y, n_epochs=50)
    print(f"Final Accuracy: {acc[-1]:.4f}, Final Loss: {loss[-1]:.4f}")
    plot_all_stats(w_mean, w_std, g_mean, g_std, loss, act)
