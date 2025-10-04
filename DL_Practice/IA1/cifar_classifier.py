from random import random

import numpy as np
import pandas as pd
import torchvision
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from torch import nn
from torch.utils.data import DataLoader, dataloader, random_split, ConcatDataset
from torchvision import transforms
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed=42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


class CNN(nn.Module):
    def __init__(self,input,num_classes,out1,out2,hidden1,kernel_size):
        super().__init__()
        self.hidden1 = hidden1
        self.conv1=nn.Conv2d(input,out1,kernel_size=kernel_size,padding='same')
        self.bn1=nn.BatchNorm2d(out1)
        self.conv2=nn.Conv2d(out1,out2,kernel_size=kernel_size,padding='same')
        self.bn2=nn.BatchNorm2d(out2)
        self.maxpool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.auto=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(out2,hidden1)
        self.fc2=nn.Linear(hidden1,num_classes)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(F.relu(self.bn2(self.conv2(x))))
        x = self.auto(x)
        x=x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model,train_loader,criterion,optimizer):
    model.train()
    losses=0
    correct=0
    total=0
    for batch,target in train_loader:
        batch,target=batch.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(batch)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        losses=losses + loss.item()*batch.size(0)

        _,predicted=torch.max(output,1)
        correct=correct + predicted.eq(target.data).cpu().sum().item()
        total=total + batch.size(0)
    losses=losses/len(train_loader.dataset)
    accuracy=correct/total
    return losses,accuracy

def test(model,test_loader,criterion):
    model.eval()
    losses=0
    correct=0
    total=0
    with torch.no_grad():
        for batch,target in test_loader:
            batch,target=batch.to(device),target.to(device)
            output=model(batch)
            loss=criterion(output,target)
            losses=losses + loss.item()*batch.size(0)

            _,predicted=torch.max(output,1)
            correct=correct + predicted.eq(target.data).cpu().sum().item()
            total=total + batch.size(0)

    losses=losses/len(test_loader.dataset)
    accuracy=correct/total
    return losses,accuracy

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])
                                ])
data='/Users/sharmishtaganesh/Desktop/DL_Practice/data/'

dataset=torchvision.datasets.CIFAR10(root=data,download=False,transform=transform,train=True)
val_size=int(len(dataset)*0.2)
train_size=len(dataset)-val_size
train_dataset,val_dataset=random_split(dataset,[train_size,val_size])
test_dataset=torchvision.datasets.CIFAR10(root=data,download=False,transform=transform)

train_loader=DataLoader(train_dataset,shuffle=True,batch_size=64)
val_loader=DataLoader(val_dataset,shuffle=False,batch_size=64)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

# mean,std,nb_samples=0,0,0
# for batch,_ in data_loader:
#     batch_size = batch.size(0)
#     nb_samples += batch_size
#     batch=batch.view(batch_size,batch.size(1),-1)
#     mean+=batch.mean(2).sum(0)
#     std+=batch.std(2).sum(0)
# mean/=nb_samples
# std/=nb_samples
# print(mean)
# print(std)

space={"out1":hp.choice("out1",[16,32,64]),
       "out2":hp.choice("out2",[32,64,128]),
       "hidden1":hp.choice("hidden1",[128,256,512]),
       "kernel_size":hp.choice("kernel_size",[3,5,7])}

def objective(params):
    model=CNN(3,10,**params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=5e-4)

    best_loss=np.inf
    best_acc=0
    patience=3
    patience_counter=0
    for epoch in range(20):
        train_losses,train_accuracy=train(model,train_loader,criterion,optimizer)
        val_losses,val_accuracy=test(model,val_loader,criterion)
        if val_losses < best_loss:
            best_loss=val_losses
            best_acc=val_accuracy
            patience_counter=0
        else:
            patience_counter+=1
            if patience_counter>=patience:
                break
    return {'loss': best_loss, 'accuracy': best_acc, 'status': STATUS_OK}

trial=Trials()
best=fmin(objective,trials=trial,space=space,algo=tpe.suggest,max_evals=10)

out1=[16,32,64][best['out1']]
out2=[32,64,128][best['out2']]
hidden1=[128,256,512][best['hidden1']]
kernel_size=[3,5,7][best['kernel_size']]

final_model=CNN(3,10,out1,out2,hidden1,kernel_size).to(device)
final_criterion = nn.CrossEntropyLoss()
final_optimizer=torch.optim.Adam(final_model.parameters(),lr=0.0001)
scheduler=torch.optim.lr_scheduler.StepLR(final_optimizer,step_size=5,gamma=0.1)

combined_dataset = ConcatDataset([train_dataset, val_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

for epoch in range(5):
    train_losses,train_accuracy=train(final_model,combined_loader,final_criterion,final_optimizer)
    scheduler.step()
    print(f"Epoch: {epoch+1}, train_loss: {train_losses:.4f}, train_accuracy: {train_accuracy:.4f}")
test_losses,test_accuracy=test(final_model,test_loader,final_criterion)
print(f"test_loss: {test_losses:.4f}, test_accuracy: {test_accuracy:.4f}")

