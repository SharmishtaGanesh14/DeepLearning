import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_size,32,kernel_size=5,padding='same') # 3, 32, 32 -> 32, 32, 32
        self.conv2 = nn.Conv2d(32,64,kernel_size=5,padding='same') # 32, 32, 32 -> 64, 32, 32
        self.maxpool = nn.MaxPool2d(2,2) # 64, 16, 16
        self.fc1 = nn.Linear(64*16*16,256)
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model,criterion,optimizer,train_loader):
    model.train()
    losses=0
    correct=0
    total=0
    for batch,target in train_loader:
        batch,target=batch.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(batch)
        _,pred=torch.max(output,1)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()

        losses += loss.item()*batch.size(0)
        total += batch.size(0)
        correct+=(pred==target).sum().item()

    losses=losses/len(train_loader.dataset)
    accuracy=correct/total
    return losses,accuracy

def test(model,criterion,test_loader):
    model.eval()
    losses=0
    correct=0
    total=0
    with torch.no_grad():
        for batch,target in test_loader:
            batch,target=batch.to(device),target.to(device)
            output=model(batch)
            _,pred=torch.max(output,1)
            loss=criterion(output,target)

            losses += loss.item()*batch.size(0)
            total += batch.size(0)
            correct+=(pred==target).sum().item()

    losses=losses/len(test_loader.dataset)
    accuracy=correct/total
    return losses,accuracy


transform=transforms.Compose([transforms.ToTensor()])
train_dataset=datasets.CIFAR10(train=True, download=True, transform=transform, root='./data')
test_dataset=datasets.CIFAR10(train=False, download=True, transform=transform, root='./data')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
images,labels=next(iter(train_loader))
print(images.shape,labels.shape)

model=CNN(3,10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    train_losses,train_accuracy=train(model,criterion,optimizer,train_loader)
    print("Epoch:",epoch+1,"Loss:",train_losses,"Accuracy:",train_accuracy)
test_losses,test_accuracy=test(model,criterion,test_loader)
print("Test Loss:",test_losses)
print("Test Accuracy:",test_accuracy)
