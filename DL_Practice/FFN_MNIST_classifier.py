import torch
from torch import nn
from torchvision import datasets, transforms
from torch.nn.functional import relu,softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
train_dataset=datasets.MNIST(train=True,transform=transform,download=False,root='./data')
test_dataset=datasets.MNIST(train=False,transform=transform,download=False,root='./data')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class FFN(nn.Module):
    def __init__(self, in_features, out_features):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x=relu(self.fc1(x))
        x=relu(self.fc2(x))
        return self.fc3(x) # raw logits - softmax done by crossentropyloss

def train(model, train_loader, criterion, optimizer):
    model.train()
    loss = 0
    correct = 0
    total = 0
    for data,target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss += loss.item()*data.size(0)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted==target).sum().item()

    loss = loss / len(train_loader.dataset)
    accuracy = correct / total
    return loss, accuracy

def test(model, test_loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_loss = criterion(output, target)
            loss += batch_loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            correct += (predicted==target).sum().item()
            total += target.size(0)

    loss = loss / len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy

model = FFN(in_features=784, out_features=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    loss, accuracy = train(model, train_loader, criterion, optimizer)
    print(f'[Training] Epoch: {epoch+1}, Loss: {loss}, Accuracy: {accuracy}')
loss, accuracy = test(model, test_loader, criterion)
print(f'[Test] Loss: {loss}, Accuracy: {accuracy}')





