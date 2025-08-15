# Date created: 15/08/2025
# Author: Sharmishta G
# Supervisor: Shyam Rajagopalan
# Aim: Build a simple deep neural network (called deep network henceforth) classifier using PyTorch.
# Train a deep network model, learn when to stop training, how to use batches and how to tune hyperparameters.

import torch.nn as nn
from scipy.special.cython_special import hyp1f1
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


def preprocessing(x,y):
    # train test split
    x_temp,x_test,y_temp,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    x_train,x_val,y_train,y_val = train_test_split(x_temp,y_temp,test_size=0.25,random_state=42)

    # scaling the dataset
    scaler_x = StandardScaler()
    x_train_scaled=scaler_x.fit_transform(x_train)
    x_val_scaled=scaler_x.transform(x_val)
    x_test_scaled=scaler_x.transform(x_test)

    # conversion to tensors
    x_train_scaled_tensor=torch.tensor(x_train_scaled,dtype=torch.float32)
    x_val_scaled_tensor=torch.tensor(x_val_scaled,dtype=torch.float32)
    x_test_scaled_tensor=torch.tensor(x_test_scaled,dtype=torch.float32)
    y_train_tensor=torch.tensor(y_train,dtype=torch.long)
    y_val_tensor=torch.tensor(y_val,dtype=torch.long)
    y_test_tensor=torch.tensor(y_test,dtype=torch.long)

    input_dim=x_train.shape[1]
    output_dim=len(set(y_train))

    # mapping x to y
    train_ds=TensorDataset(x_train_scaled_tensor,y_train_tensor)
    val_ds=TensorDataset(x_val_scaled_tensor,y_val_tensor)
    test_ds=TensorDataset(x_test_scaled_tensor,y_test_tensor)

    return train_ds,val_ds,test_ds,input_dim,output_dim

x,y=make_classification(n_samples=3000, n_features=50, n_classes=2, random_state=42)
train_ds,val_ds,test_ds,inp_dim,op_dim=preprocessing(x,y)
batchsize=64
train_dataloader = DataLoader(train_ds,batch_size=batchsize,shuffle=True)
val_dataloader = DataLoader(val_ds,batch_size=batchsize)
test_dataloader = DataLoader(test_ds,batch_size=batchsize)

class FFN(nn.Module):
    def __init__(self,input_dim,hd1,hd2,output_dim,dropout=0.2):
        super().__init__()
        self.input_dim=input_dim
        self.hd1=hd1
        self.hd2=hd2
        self.output_dim=output_dim
        self.network=nn.Sequential(
            nn.Linear(input_dim,hd1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hd1,hd2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hd2,output_dim)
        )

    def forward(self,x):
        return self.network(x)

h1,h2,dropout,lr,epochs,patience,patience_counter=64,64,0.2,0.0001,100,10,0
correct=0
total=0
best_val_accuracy=0
best_state=None

model = FFN(inp_dim, h1, h2, op_dim, dropout)
optimiser=torch.optim.Adam(model.parameters(),lr=lr,)
loss_fn=nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for batch_x,batch_y in train_dataloader:
        optimiser.zero_grad()
        logits=model(batch_x)
        loss=loss_fn(logits,batch_y)
        loss.backward()
        optimiser.step()

    model.eval()
    with torch.no_grad():
        for batch_x,batch_y in val_dataloader:
            logits=model(batch_x)
            pred=torch.argmax(logits,dim=1)
            correct+=torch.sum(pred==batch_y).item()
            total+=batch_y.shape[0]
    val_accuracy=correct/total
    print(f"Epoch {epoch + 1}: Validation Accuracy = {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy=val_accuracy
        best_state=model.state_dict()
        patience_counter=0
    else:
        patience_counter+=1
    if patience_counter>patience:
        print(f"Early stopping at epoch {epoch + 1}")
        model.load_state_dict(best_state)
        break

print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

# Final test evaluation (completely unseen data)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for Xb, yb in test_dataloader:
        logits = model(Xb)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
test_acc = correct / total
print(f"Final Test Accuracy on Unseen Data: {test_acc:.4f}")