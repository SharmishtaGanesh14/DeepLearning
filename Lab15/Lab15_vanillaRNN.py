import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# dataset class
class ImageCaptionDataset(Dataset):
    def __init__(self, merged_df):
        self.df = merged_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_feat = torch.tensor(row["features"], dtype=torch.float32)
        caption_tokens = torch.tensor(row["caption_tokens"], dtype=torch.long)
        return img_feat, caption_tokens

# collate function to pad captions
def collate_fn(batch, vocab):
    img_feats, captions = zip(*batch)
    img_feats = torch.stack(img_feats, 0)  # [batch, feat_size]
    captions = pad_sequence(captions, batch_first=True, padding_value=vocab["<PAD>"])  # [batch, max_seq_len]
    return img_feats, captions

# RNN model
class ImageCaptionRNN(nn.Module):
    def __init__(self, img_feat_size, embed_size, hidden_size, vocab_size, embedding_matrix):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.hidden_init = nn.Linear(img_feat_size, hidden_size)  # init hidden from image
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.start_token_idx = 1  # index of <START> in vocab
        self.hidden_size = hidden_size

    def forward(self, img_feat, max_len=20):
        """Autoregressive caption generation"""
        batch_size = img_feat.size(0)
        h0 = torch.tanh(self.hidden_init(img_feat)).unsqueeze(0)  # [1, batch, hidden_size]

        start_idx = torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=img_feat.device)
        input_t = self.embed(start_idx).unsqueeze(1)  # [batch, 1, embed_size]

        outputs, hidden = [], h0
        for _ in range(max_len):
            out, hidden = self.rnn(input_t, hidden)       # [batch, 1, hidden_size]
            logits = self.fc(out.squeeze(1))              # [batch, vocab_size]
            outputs.append(logits)
            predicted_token = logits.argmax(dim=-1)       # greedy prediction
            input_t = self.embed(predicted_token).unsqueeze(1)

        return torch.stack(outputs, dim=1)  # [batch, seq_len, vocab_size]

def main():
    # Load vocab and data
    with open("DataAndProcessing/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("DataAndProcessing/inv_vocab.pkl", "rb") as f:
        inv_vocab = pickle.load(f)

    ground_truth = pd.read_pickle("DataAndProcessing/ground_truth.pkl")
    features_df = pd.read_pickle("DataAndProcessing/features_to_rnn.pkl")
    merged_df = pd.merge(ground_truth, features_df, on="image")
    print(f"Merged dataset size: {len(merged_df)}")

    # Split into train/test
    train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
    train_dataset = ImageCaptionDataset(train_df)
    test_dataset = ImageCaptionDataset(test_df)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, vocab))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, vocab))

    embedding_matrix = np.load("DataAndProcessing/embedding_matrix.npy")
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    vocab_size, embed_size = embedding_matrix.shape

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 512
    model = ImageCaptionRNN(
        img_feat_size=2048,
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])  # ignore padding

    num_epochs = 5

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for img_feat, caption_tokens in train_loader:
            img_feat, caption_tokens = img_feat.to(device), caption_tokens.to(device)
            optimizer.zero_grad()
            outputs = model(img_feat, max_len=caption_tokens.size(1))  # [batch, seq_len, vocab_size]

            loss = sum(
                criterion(outputs[:, t, :], caption_tokens[:, t])
                for t in range(caption_tokens.size(1))
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Inference with Corpus BLEU
    model.eval()
    total_loss, references, hypotheses = 0, [], []
    smooth_fn = SmoothingFunction().method1

    with torch.no_grad():
        for img_feat, caption_tokens in test_loader:
            img_feat, caption_tokens = img_feat.to(device), caption_tokens.to(device)
            batch_size, seq_len = img_feat.size(0), caption_tokens.size(1)

            # Forward pass for loss
            outputs = model(img_feat, max_len=seq_len)

            # Compute loss
            loss = sum(
                criterion(outputs[:, t, :], caption_tokens[:, t])
                for t in range(seq_len)
            )
            total_loss += loss.item()

            # Greedy decoding for captions
            pred_ids = outputs.argmax(dim=-1).cpu().numpy()  # [batch, seq_len]
            gt_ids = caption_tokens.cpu().numpy()

            for b in range(batch_size):
                # predicted
                pred_caption = []
                for idx in pred_ids[b]:
                    token = inv_vocab.get(idx, "<UNK>")
                    if token == "<END>":
                        break
                    pred_caption.append(token)

                # ground truth
                gt_caption = []
                for idx in gt_ids[b]:
                    token = inv_vocab.get(idx, "<UNK>")
                    if token == "<END>":
                        break
                    gt_caption.append(token)

                hypotheses.append(pred_caption)
                references.append([gt_caption])  # corpus_bleu expects list of references per sentence

    # Final scores
    avg_loss = total_loss / len(test_loader)
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smooth_fn)

    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Corpus BLEU Score: {bleu_score:.4f}")

if __name__ == '__main__':
    main()

