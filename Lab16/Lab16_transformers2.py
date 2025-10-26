import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import spacy
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from gensim.models import KeyedVectors
from tqdm import tqdm

MAX_LEN = 40

# ---------------- Utilities ----------------
def tokenize(text, tok):
    return [t.text for t in tok(text)]

def build_vocab(texts, tok, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text, tok))
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    inv_vocab = {i: w for w, i in vocab.items()}
    return vocab, inv_vocab

def encode(text, vocab, tok):
    tokens = ["<sos>"] + tokenize(text, tok) + ["<eos>"]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    if len(ids) < MAX_LEN:
        ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return ids

# ---------------- Dataset ----------------
class TranslationDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = torch.tensor(src, dtype=torch.long)
        self.tgt = torch.tensor(tgt, dtype=torch.long)
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

# ---------------- Model ----------------
class TransformerMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size=256, nhead=8, num_layers=4, dim_ff=512, dropout=0.1):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, emb_size)
        self.pos_encoder = nn.Embedding(MAX_LEN, emb_size)
        self.pos_decoder = nn.Embedding(MAX_LEN, emb_size)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src, tgt):
        seq_len_src, seq_len_tgt = src.size(1), tgt.size(1)
        pos_src = torch.arange(seq_len_src).unsqueeze(0).to(src.device)
        pos_tgt = torch.arange(seq_len_tgt).unsqueeze(0).to(tgt.device)

        src_emb = self.src_embed(src) + self.pos_encoder(pos_src)
        tgt_emb = self.tgt_embed(tgt) + self.pos_decoder(pos_tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_tgt).to(src.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)

# ---------------- Training / Evaluation ----------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, tgt_inv, device):
    model.eval()
    total_loss = 0
    chencherry = SmoothingFunction()
    candidates, references = [], []

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

            pred_tokens = output.argmax(-1).cpu().numpy()
            for i in range(len(pred_tokens)):
                pred = [tgt_inv[idx] for idx in pred_tokens[i] if idx not in [0,1,2,3]]
                true = [tgt_inv[idx.item()] for idx in tgt[i] if idx.item() not in [0,1,2,3]]
                if pred and true:
                    candidates.append(pred)
                    references.append([true])

    bleu = corpus_bleu(references, candidates, smoothing_function=chencherry.method4)
    return total_loss / len(dataloader), bleu

def translate_sentence(model, sentence, src_vocab, tgt_vocab, tgt_inv, hi_tok, device):
    model.eval()
    src = torch.tensor([encode(sentence, src_vocab, hi_tok)], dtype=torch.long).to(device)
    tgt = torch.tensor([[tgt_vocab["<sos>"]]], dtype=torch.long).to(device)
    for _ in range(MAX_LEN):
        out = model(src, tgt)
        next_token = out[:, -1, :].argmax(-1).unsqueeze(0)
        tgt = torch.cat([tgt, next_token], dim=1)
        if next_token.item() == tgt_vocab["<eos>"]:
            break
    tokens = [tgt_inv[idx.item()] for idx in tgt[0][1:] if idx.item() != tgt_vocab["<eos>"]]
    return " ".join(tokens)

# ---------------- Main Pipeline ----------------
def main():
    # Load data
    data = pd.read_csv("Hindi_English_Truncated_Corpus.csv")
    data = data.rename(columns={"english_sentence": "en", "hindi_sentence": "hi"}).dropna()
    data["en"] = data["en"].str.lower().str.strip()
    data["hi"] = data["hi"].str.lower().str.strip()

    # Tokenizers
    en_tok = spacy.load("en_core_web_sm")
    hi_tok = spacy.blank("hi")

    # Build vocab
    src_vocab, src_inv = build_vocab(data["hi"], hi_tok)
    tgt_vocab, tgt_inv = build_vocab(data["en"], en_tok)

    # Encode sentences
    X = [encode(x, src_vocab, hi_tok) for x in data["hi"]]
    Y = [encode(y, tgt_vocab, en_tok) for y in data["en"]]

    # Train-validation split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    train_loader = DataLoader(TranslationDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TranslationDataset(X_val, Y_val), batch_size=32)

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerMT(len(src_vocab), len(tgt_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])

    # Load pre-trained embeddings
    ft = KeyedVectors.load_word2vec_format('cc.hi.300.vec', limit=100000)
    with torch.no_grad():
        for word, idx in src_vocab.items():
            if word in ft:
                model.src_embed.weight[idx] = torch.tensor(ft[word])

    # Training loop
    for epoch in range(20):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, bleu = evaluate(model, val_loader, criterion, tgt_inv, device)
        print(f"Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | BLEU: {bleu:.4f}")

    torch.save(model.state_dict(), "custom_transformer_en_hi.pth")
    print("Model saved successfully!")

    # Translate example
    print(translate_sentence(model, "यह एक नई वाक्य है।", src_vocab, tgt_vocab, tgt_inv, hi_tok, device))

if __name__ == "__main__":
    main()
