import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from torch import nn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# [optimised version of lab10 using hyperopt]
class GeneExpressionDataset:
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float).to(device)
        self.y = torch.tensor(y,dtype=torch.float).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_clean_dataset(path1,path2):
    lm=pd.read_csv(path1,header=0).iloc[:,4:]
    tg=pd.read_csv(path2,header=0).iloc[:,4:]

    X=lm.apply(pd.to_numeric).T.values
    Y=tg.apply(pd.to_numeric).T.values

    x_train,x_rest,y_train,y_rest=train_test_split(X,Y,test_size=0.3,random_state=42)
    x_test,x_val,y_test,y_val=train_test_split(x_rest,y_rest,test_size=0.5,random_state=42)

    scaler_x=StandardScaler()
    x_train=scaler_x.fit_transform(x_train)
    x_test=scaler_x.transform(x_test)
    x_val=scaler_x.transform(x_val)

    scaler_y=StandardScaler()
    y_train=scaler_y.fit_transform(y_train)
    y_test=scaler_y.transform(y_test)
    y_val=scaler_y.transform(y_val)

    return (GeneExpressionDataset(x_train,y_train),GeneExpressionDataset(x_val,y_val), GeneExpressionDataset(x_test,y_test), X.shape[1],Y.shape[1])

class FFN(nn.Module):
    def __init__(self,input_dim,output_dim,h1,h2,dr):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.h1 = h1
        self.h2 = h2
        self.dr = dr
        self.network=nn.Sequential(
            nn.Linear(self.input_dim,self.h1),
            nn.Dropout(dr),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.Dropout(dr),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Linear(h2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def train(dataloader,model,loss_fn,optimizer,epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*data.size(0)
    loss_avg = total_loss / len(dataloader.dataset)
    return loss_avg

def test(dataloader,model,loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()*data.size(0)
        loss_avg = total_loss / len(dataloader.dataset)
        return loss_avg

def main():


    train_ds,test_ds,val_ds,input_layer,op_layer=load_and_clean_dataset("1000G_landmark_genes.csv","1000G_target_genes.csv")
    train_data_loader=DataLoader(train_ds,batch_size=64,shuffle=True)
    test_data_loader=DataLoader(test_ds,batch_size=64,shuffle=False)
    val_data_loader=DataLoader(val_ds,batch_size=64,shuffle=False)

    loss_fn = nn.MSELoss()

    space = {
        "hidden1": hp.choice("hidden1", [128, 256, 512]),
        "hidden2": hp.choice("hidden2", [64, 128, 256]),
        "dropout": hp.uniform("dropout", 0.1, 0.5),
        "lr": hp.loguniform("lr", -9, -4)  # ~1e-4 to 1e-2
    }

    def objective(params):
        model = FFN(input_layer, op_layer,
                    h1=params["hidden1"],
                    h2=params["hidden2"],
                    dr=params["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=1e-5)

        # Short training for hyperopt
        for epoch in range(30):
            train(train_data_loader, model, loss_fn, optimizer,epoch+1)

        val_loss = test(val_data_loader, model, loss_fn)
        return {"loss": val_loss, "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=25, trials=trials)
    print("\nBest hyperparameters found by Hyperopt:")
    print(best)

    h1_candidates = [128, 256, 512]
    h2_candidates = [64, 128, 256]

    h1 = h1_candidates[best["hidden1"]]
    h2 = h2_candidates[best["hidden2"]]
    dr = best["dropout"]
    lr = best["lr"]

    print(f"\nUsing: hidden1={h1}, hidden2={h2}, dropout={dr:.3f}, lr={lr:.6f}")

    final_model = FFN(input_layer, op_layer, h1, h2, dr).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=1e-5)

    epochs = 300
    train_losses = []

    # Combine train+val for final training
    train_val_loader = DataLoader(ConcatDataset([train_ds, val_ds]), batch_size=32, shuffle=True)

    for t in range(epochs):
        train_loss = train(train_val_loader, final_model, loss_fn, final_optimizer, t+1)
        train_losses.append(train_loss)
        if (t + 1) % 50 == 0:
            print(f"Epoch {t + 1} | Train Loss: {train_loss:.4f}")

    test_loss = test(test_data_loader, final_model, loss_fn)
    print(f"\nFinal Test Loss: {test_loss:.4f}")

    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    # Save model
    torch.save(final_model.state_dict(), "tg_prediction_from_lm_genes.pth")
    print("Saved final model to tg_prediction_from_lm_genes.pth")


if __name__ == '__main__':
    main()














