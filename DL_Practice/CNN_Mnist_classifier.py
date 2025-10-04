
import torch
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from torch import nn
from torchvision import datasets, transforms
from torch.nn.functional import relu

data='./data'
random_seed=42
torch.manual_seed(random_seed)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=64
no_of_classes=10
epochs=5
epochs2=5

class CNN(nn.Module):
    def __init__(self,fc1,fc2,outchannels1,outchannels2,kernel_size):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=outchannels1,kernel_size=kernel_size,padding='same') #1, 28, 28 -> 32, 28, 28
        self.conv2=nn.Conv2d(in_channels=outchannels1,out_channels=outchannels2,kernel_size=kernel_size,padding='same') #32, 28, 28 -> 64, 28, 28
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2) #32, 28, 28 -> 32, 14, 14
        self.l1=nn.Linear(in_features=outchannels2*7*7,out_features=fc1)
        self.l2=nn.Linear(in_features=fc1,out_features=fc2)
        self.l3=nn.Linear(in_features=fc2,out_features=no_of_classes)

    def forward(self,x):
        x=self.maxpool(relu(self.conv1(x))) # 32, 14, 14
        x=self.maxpool(relu(self.conv2(x)))  # 64, 7, 7
        x=relu(self.l1(x.view(x.size(0),-1))) # 64*7*7 -> fc1
        x=relu(self.l2(x)) # fc1 -> fc2
        x=self.l3(x)# fc3->10
        return x

def train(model,loader,criterion,optimizer):
    model.train()
    losses=0
    correct=0
    total=0
    for data,target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output=model(data)
        _,pred=torch.max(output,1)
        loss=criterion(output,target)
        correct += (pred==target).sum().item()
        total += target.size(0)
        loss.backward()
        optimizer.step()
        losses=losses + loss.item() * data.size(0)
    losses=losses/len(loader.dataset)
    accuracy=(correct/total)*100
    return losses,accuracy

def evaluate(model,loader,criterion):
    model.eval()
    losses=0
    correct=0
    total=0
    with torch.no_grad():
        for data,target in loader:
            data, target = data.to(device), target.to(device)
            output=model(data)
            _,pred=torch.max(output,1)
            loss=criterion(output,target)
            losses=losses + loss.item() * data.size(0)
            correct += (pred == target).sum().item()
            total += target.size(0)
    losses=losses/len(loader.dataset)
    accuracy=(correct/total)*100
    return losses,accuracy

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,),(0.3081,))])
dataset=datasets.MNIST(root=data,train=True,transform=transform,download=True)
test_dataset=datasets.MNIST(root=data,train=False,transform=transform,download=True)
test_size = int(len(dataset) * 0.2)
train_size = len(dataset) - test_size
train_dataset,val_dataset=torch.utils.data.random_split(dataset,[train_size,test_size])

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
val_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
train_val_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

space={
    "fc1":hp.choice("fc1",[64,128,256]),
    "fc2":hp.choice("fc2",[64,128,256]),
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-2)),
    "outchannels1":hp.choice("outchannels1",[16,32,64,128]),
    "outchannels2":hp.choice("outchannels2",[16,32,64,128]),
    "kernel_size":hp.choice("kernel_size",[3,5,7])
}

def objective(params):
    lr = params["lr"]
    model_params = {k: v for k, v in params.items() if k != "lr"}
    model = CNN(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    patience = 3

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return {
        "loss": best_val_loss,
        "accuracy": best_val_acc,
        "status": STATUS_OK
    }


trials = Trials()
best=fmin(objective,space=space,algo=tpe.suggest,trials=trials,max_evals=25)

h1 = [64,128,256][best["fc1"]]
h2 = [64,128,256][best["fc2"]]
outchannels1 = [16,32,64,128][best["outchannels1"]]
outchannels2 = [16,32,64,128][best["outchannels2"]]
kernel_size = [3,5,7][best["kernel_size"]]
lr = best["lr"]
print("Best hyperparameters:", {
    "fc1": h1,
    "fc2": h2,
    "outchannels1": outchannels1,
    "outchannels2": outchannels2,
    "kernel_size": kernel_size,
    "lr": lr
})

final_model = CNN(h1,h2,outchannels1,outchannels2,kernel_size).to(device)
optimizer_final = torch.optim.Adam(final_model.parameters(), lr=lr)
final_criterion = nn.CrossEntropyLoss()
for epoch in range(epochs2):
    train_loss, train_accuracy = train(final_model, train_val_loader, final_criterion, optimizer_final)
    print(f"[Final Model] Epoch {epoch + 1}/{epochs2} - Loss: {train_loss:.4f}, Acc: {train_accuracy:.2f}%")
test_loss, test_accuracy = evaluate(final_model, test_loader, final_criterion)
print("Final test loss:", test_loss, "Final test accuracy:", test_accuracy)

