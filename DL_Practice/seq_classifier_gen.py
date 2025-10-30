import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

device='cuda' if torch.cuda.is_available() else 'cpu'

class DNA_Sequence_Dataset(Dataset):
    def __init__(self,sequences,labels):
        super(DNA_Sequence_Dataset, self).__init__()
        self.sequences = sequences
        self.labels = labels
        self.mapping={'A':0,'T':1,'G':2,'C':3}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        label = self.labels[index]
        mapped = torch.tensor([self.mapping[w] for w in seq.strip().upper()], dtype=torch.long)
        encoded = F.one_hot(mapped, num_classes=4).float()
        return encoded, label

class GRU(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_dim=2):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_dim)

    def forward(self,x):
        _,h=self.gru(x)
        h_last = h[-1]
        out=self.fc(h_last)
        return out

def train(train_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def test(test_loader, model, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, dim=1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def pad_collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels=torch.tensor(labels,dtype=torch.long)
    return padded_sequences, labels

# class GRU_classifier
base_sequences = [
            "CATGCATGACTT", "CGTAGCTGAGCA",
            "GCATGCAGCTTA", "TGCATGCATGCA",
            "GCTACGTAGGCA", "GTACGTAGCTA"
        ]
base_labels = [1, 1, 0, 1, 0, 1]
data,targets=[],[]
for i in range(600):
    for label,seq in zip(base_labels,base_sequences):
        data.append(seq)
        targets.append(label)

dataset=DNA_Sequence_Dataset(data,targets)
train_len=int(len(dataset)*0.8)
test_len=len(dataset)-train_len
train_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_len,test_len])

train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,collate_fn=pad_collate_fn)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,collate_fn=pad_collate_fn)

vocab_size=4
embed_dim=4
hidden_dim=64
output_dim=2

model=GRU(input_size=embed_dim,hidden_size=hidden_dim,num_layers=4,output_dim=output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

for epoch in range(10):
    train_loss, train_acc = train(train_loader, model, criterion, optimizer)
    test_loss, test_acc = test(test_loader, model, criterion)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
    scheduler.step()