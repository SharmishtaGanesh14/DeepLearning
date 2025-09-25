import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# bleu

# Dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_feat = torch.tensor(row["features"], dtype=torch.float32)
        cap_embed = torch.tensor(row["embedding"], dtype=torch.float32)
        return img_feat, cap_embed

# Model
class ImageCaptionRNN(nn.Module):
    def __init__(self, img_feat_size=2048, hidden_size=256, embed_size=50, num_layers=1):
        super().__init__()
        self.img_fc = nn.Linear(img_feat_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_size)

    def forward(self, img_feats):
        x = self.img_fc(img_feats)         # [batch, seq_len=1, embed_size]
        hiddens, _ = self.rnn(x)                  # [batch, 1, hidden_size]
        out = self.linear(hiddens)                # [batch, 1, embed_size]
        return out.squeeze(1)                     # [batch, embed_size]

def main():
    # Load data
    data = pd.read_pickle("DataAndProcessing/ground_truth.pkl")
    features = pd.read_pickle("DataAndProcessing/features_to_rnn.pkl")
    merged = pd.merge(data, features, on="image")

    train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)
    train_dataset = ImageCaptionDataset(train_df)
    test_dataset = ImageCaptionDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model = ImageCaptionRNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    EPOCHS = 5
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img_feats, cap_embeds in train_loader:
            img_feats, cap_embeds = img_feats.to(device), cap_embeds.to(device)
            outputs = model(img_feats)
            loss = criterion(outputs, cap_embeds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for img_feats, cap_embeds in test_loader:
            img_feats, cap_embeds = img_feats.to(device), cap_embeds.to(device)
            outputs = model(img_feats)
            loss = criterion(outputs, cap_embeds)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()