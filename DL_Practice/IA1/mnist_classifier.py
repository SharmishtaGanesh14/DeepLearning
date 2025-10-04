import torch
from torch import nn
from torchvision import transforms, datasets
import numpy as np

seed=44
torch.manual_seed(seed)
np.random.seed(seed)
data='/Users/sharmishtaganesh/Desktop/DL_Practice/data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        layers=[]
        layers.append(nn.Linear(in_features, 128))
        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(128, 64))
        layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(64, out_features)) # sigmoid as a part of loss
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        x=x.view(x.size(0), -1)
        return self.network(x)

def Train(model,loader,criterion,optimizer):
    model.train()
    losses=0
    correct=0
    total=0
    for batch,target in loader:
        batch,target=batch.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(batch)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        losses=losses + loss.item()*batch.size(0)

        _, predicted = torch.max(output.data, 1)
        correct=correct + predicted.eq(target.data).cpu().sum().item()
        total=total + batch.size(0)

    losses=losses/len(loader.dataset)
    accuracy=correct/total
    return losses, accuracy

def test(model,loader,criterion):
    model.eval()
    losses=0
    correct=0
    total=0
    with torch.no_grad():
        for batch,target in loader:
            batch,target=batch.to(device),target.to(device)
            output=model(batch)
            loss=criterion(output,target)
            losses=losses + loss.item()*batch.size(0)

            _, predicted = torch.max(output.data, 1)
            correct=correct + predicted.eq(target.data).cpu().sum().item()
            total=total + batch.size(0)
    accuracy=correct/total
    losses=losses/len(loader.dataset)
    return losses, accuracy

transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.1307], std=[0.3081])
                            ])

train_dataset=datasets.MNIST(root=data, train=True, transform=transform, download=False)
test_dataset=datasets.MNIST(root=data, train=False, transform=transform, download=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# mean,std,nb_smaple=0,0,0
# for batch_idx, (data, target) in enumerate(train_loader):
#     batch = data.size(0)
#     data=data.view(batch,data.size(1),-1)
#     mean+=data.mean(2).sum(0).item()
#     std+=data.std(2).sum(0).item()
#     nb_smaple+=batch
# mean/=nb_smaple
# std/=nb_smaple
# print(mean)
# print(std)

# images,labels=next(iter(train_loader))
# print(images.shape)

model = FFN(in_features=784, out_features=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(5):
    losses, accuracy = Train(model,train_loader,criterion,optimizer)
    scheduler.step()
    print(f"Train Loss: {losses}, Train Accuracy: {accuracy}")

test_losses, test_accuracy = test(model,test_loader,criterion)
print(f"Test Loss: {test_losses}, Test Accuracy: {test_accuracy}")

