import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset

torch.manual_seed(42)
x=torch.randn(1024,100)
y=torch.randn(1024,1)
input_size=100
hidden1=64
hidden2=32
output_size=1
batch_size=64
learning_rate=0.01
epochs=10

dataset=TensorDataset(x,y)
dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

model=nn.Sequential(nn.Linear(input_size,hidden1),
                    nn.ReLU(),
                    nn.Linear(hidden1,hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2,output_size))
criterion = nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate) # optimiser that we wabt to use


# training loop
# nn base core module
for epoch in range(epochs):
    epoch_loss=0
    for batch_x,batch_y in dataloader:
        # forward pass
        predictions=model(batch_x)
        loss=criterion(predictions,batch_y)

        optimizer.zero_grad() # clear the gradients before the next batch update
        loss.backward()  # compute the gradients - even when I don't mention the formula it will compute
        # - Autograd - compute the gradient normally
        optimizer.step()   # update the weights
        epoch_loss+=loss.item()
    avg_loss=epoch_loss/len(dataloader)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.3f}')


# requires_grad - telling this variable needs gradient computation
# all the hyperparameters are defined implicitly, but also can be set manually

# implement He init
# batch norm
# dropout


# performance metrics
# COMPLETE LAB 10

