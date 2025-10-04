import torch
from torch import nn
from torchvision import models, datasets, transforms

# Config
data = '../data/'
batch_size = 64
random_seed = 654
torch.manual_seed(random_seed)

lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_decay = 5e-4
epochs = 5
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Data loading
def load_process_data(transform):
    train_dataset = datasets.CIFAR10(train=True, download=True, transform=transform, root=data)
    test_dataset = datasets.CIFAR10(train=False, download=True, transform=transform, root=data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -------------------
# Training
# -------------------
def train_model(epoch, train_loader, model, loss_fn, optimiser):
    model.train()
    loss_total = 0.0
    correct, total = 0, 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimiser.step()

        # accumulate for averages
        loss_total += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}, '
                  f'Accuracy: {correct/total:.6f}')

    avg_loss = loss_total / len(train_loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Testing
def test_model(epoch, test_loader, model, loss_fn):
    model.eval()
    loss_total = 0.0
    correct, total = 0, 0
    correct_dict = {classname: 0 for classname in classes}
    total_dict = {classname: 0 for classname in classes}

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            # accumulate
            loss_total += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            # per-class stats
            for pred, tg in zip(predicted, target):
                tg = tg.item()
                pred = pred.item()
                if pred == tg:
                    correct_dict[classes[tg]] += 1
                total_dict[classes[tg]] += 1

            if batch_idx % 100 == 0:
                print(f'Test Epoch: {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}, '
                      f'Accuracy: {correct}/{total}')

    avg_loss = loss_total / len(test_loader.dataset)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, correct_dict, total_dict

# Main
def main():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
    model = model.to(device)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    print(weights.transforms())
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader, test_loader = load_process_data(transform)

    # Loss + Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    # Train
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_model(epoch, train_loader, model, loss_fn, optimiser)
        print(f"--------------- Epoch {epoch}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}% ----------------")

    # Final test
    test_loss, test_acc, correct_dict, total_dict = test_model(epochs, test_loader, model, loss_fn)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n")
    for classname in classes:
        print(f"{classname}: {correct_dict[classname]}/{total_dict[classname]}")

if __name__ == '__main__':
    main()
