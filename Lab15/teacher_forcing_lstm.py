import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
from PIL import Image
import random


# ---------------- Dataset & Dataloader ----------------
class ImageCaptionDataset(Dataset):
    def __init__(self, merged_df):
        self.df = merged_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_feat = torch.tensor(row["features"], dtype=torch.float32)
        caption_tokens = torch.tensor(row["caption_tokens"], dtype=torch.long)
        return img_feat, caption_tokens, row["image"]  # include image path


def collate_fn(batch):
    img_feats, captions, img_paths = zip(*batch)
    img_feats = torch.stack(img_feats, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab["<PAD>"])
    return img_feats, captions, img_paths


# ---------------- Model ----------------
class ImageCaptionLSTM(nn.Module):
    def __init__(self, img_feat_size, embed_size, hidden_size, vocab_size, embedding_matrix, start_token_idx=1):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_init = nn.Linear(img_feat_size, hidden_size)
        self.start_token_idx = start_token_idx
        self.hidden_size = hidden_size

    def forward(self, img_feat, captions=None, max_len=None, teacher_forcing=True):
        batch_size = img_feat.size(0)
        device = img_feat.device

        h0 = torch.tanh(self.hidden_init(img_feat)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        hidden = (h0, c0)

        if captions is not None and teacher_forcing:
            embeds = self.embed(captions)
            out, _ = self.lstm(embeds, hidden)
            logits = self.fc(out)
            return logits
        else:
            start_idx = torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=device)
            input_t = self.embed(start_idx).unsqueeze(1)
            outputs = []
            for _ in range(max_len):
                out, hidden = self.lstm(input_t, hidden)
                logits = self.fc(out.squeeze(1))
                outputs.append(logits)
                predicted_token = logits.argmax(dim=-1)
                input_t = self.embed(predicted_token).unsqueeze(1)
            return torch.stack(outputs, dim=1)

if __name__ == "__main__":
    # ---------------- Load data ----------------
    with open("DataAndProcessing/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("DataAndProcessing/inv_vocab.pkl", "rb") as f:
        inv_vocab = pickle.load(f)

    ground_truth = pd.read_pickle("DataAndProcessing/ground_truth.pkl")
    features_df = pd.read_pickle("DataAndProcessing/features_to_rnn.pkl")
    merged_df = pd.merge(ground_truth, features_df, on="image")

    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
    train_dataset = ImageCaptionDataset(train_df)
    test_dataset = ImageCaptionDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    embedding_matrix = torch.tensor(np.load("DataAndProcessing/embedding_matrix.npy"), dtype=torch.float32)
    vocab_size, embed_size = embedding_matrix.shape

    # ---------------- Setup model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptionLSTM(2048, embed_size, 512, vocab_size, embedding_matrix).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    num_epochs = 100

    # ---------------- Training ----------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for img_feat, captions, _ in train_loader:
            img_feat, captions = img_feat.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(img_feat, captions=captions, teacher_forcing=True)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # ---------------- Save model ----------------
    torch.save(model.state_dict(), "DataAndProcessing/image_caption_lstm.pth")
    torch.save(optimizer.state_dict(), "DataAndProcessing/optimizer.pth")
    print("Model and optimizer saved successfully.")

    # ---------------- Evaluation ----------------
    model.eval()
    smooth_fn = SmoothingFunction().method1
    bleu_scores = []

    with torch.no_grad():
        for img_feat, captions, _ in test_loader:
            img_feat, captions = img_feat.to(device), captions.to(device)
            batch_size, seq_len = captions.size()
            outputs = model(img_feat, max_len=seq_len, teacher_forcing=False)
            predicted_ids = outputs.argmax(dim=-1).cpu().numpy()
            for b in range(batch_size):
                gt_tokens = [inv_vocab.get(idx, "<UNK>") for idx in captions[b].cpu().numpy()]
                if "<END>" in gt_tokens:
                    gt_tokens = gt_tokens[:gt_tokens.index("<END>")]
                pred_tokens = [inv_vocab.get(idx, "<UNK>") for idx in predicted_ids[b]]
                if "<END>" in pred_tokens:
                    pred_tokens = pred_tokens[:pred_tokens.index("<END>")]
                bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smooth_fn)
                bleu_scores.append(bleu)

    print(f"Average BLEU score: {np.mean(bleu_scores):.4f}")


    # ---------------- Caption Generation ----------------
    def generate_caption(model, img_feat, max_len, start_token_idx=1):
        model.eval()
        device = img_feat.device
        outputs = model(img_feat.unsqueeze(0), max_len=max_len, teacher_forcing=False)
        predicted_ids = outputs.argmax(dim=-1).squeeze(0).cpu().numpy()
        caption_tokens = []
        for idx in predicted_ids:
            word = inv_vocab.get(idx, "<UNK>")
            if word == "<END>":
                break
            caption_tokens.append(word)
        return " ".join(caption_tokens)


    # ---------------- Visualization ----------------
    img_dir = "DataAndProcessing/Images"  # folder containing all images
    N = 5
    sample_indices = random.sample(range(len(test_df)), N)
    plt.figure(figsize=(15, N * 3))

    for i, idx in enumerate(sample_indices):
        row = test_df.iloc[idx]
        img_path = os.path.join(img_dir, row["image"])

        # Skip missing images
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}, skipping...")
            continue

        gt_tokens = [inv_vocab.get(tok, "<UNK>") for tok in row["caption_tokens"]]
        if "<END>" in gt_tokens:
            gt_tokens = gt_tokens[:gt_tokens.index("<END>")]
        gt_caption = " ".join(gt_tokens)

        img = Image.open(img_path).convert("RGB")
        plt.subplot(N, 1, i + 1)
        plt.imshow(img)
        plt.axis("off")

        img_feat_tensor = torch.tensor(row["features"], dtype=torch.float32).to(device)
        pred_caption = generate_caption(model, img_feat_tensor, max_len=len(gt_tokens))
        plt.title(f"GT: {gt_caption}\nPred: {pred_caption}", fontsize=12)

    plt.tight_layout()
    plt.show()
