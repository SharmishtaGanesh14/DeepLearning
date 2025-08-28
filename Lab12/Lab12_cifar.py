# Date created: 28/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Implement CNN using PyTorch for image classification using cifar10 dataset
# Plot train error vs increasing number of layers

import torchvision
import torchvision.transforms as transforms
from regex import F
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.optim as optim

#load dataset without normalization to calculate mean and std
data_for_stats=torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(),download=True)
loader_for_stats=DataLoader(data_for_stats,batch_size=5000,shuffle=False)

#calculate mean and std over the entire training data
mean = 0
std = 0
nb_samples = 0
for data,target in loader_for_stats:
    batch_size = data.size(0)
    data=data.view(batch_size,data.size(1),-1)
    mean+=data.mean(2).sum(0)
    std+=data.std(2).sum(0)
    nb_samples+=batch_size
mean/=nb_samples
std/=nb_samples

mean = mean.numpy()
std = std.numpy()
print('calculated mean:', mean)
print('calculated std:', std)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)])
train_data=torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_data=torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

# function to show images
def imshow(img):
    img = img /2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# Save the trained model
PATH = 'cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Test & evaluate
def evaluate_model():
    # Load model (optional if continuing)
    net_eval = Net().to(device)
    net_eval.load_state_dict(torch.load(PATH))
    net_eval.eval()

    # Show some test images and predictions
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    imshow(torchvision.utils.make_grid(images.cpu()))

    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

    outputs = net_eval(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(predicted))))

    # Overall accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net_eval(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')

    # Accuracy per class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net_eval(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class {classname:5s}: {accuracy:.1f} %')

# Run evaluation
evaluate_model()