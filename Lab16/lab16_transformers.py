import math

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from collections import Counter
import spacy
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
MAX_LEN = 20

class TranslationDataset(Dataset):
    def __init__(self, csv_file, src_col='hindi_sentence', trg_col='english_sentence', max_len=MAX_LEN):
        self.data = pd.read_csv(csv_file)
        self.src_col = src_col
        self.trg_col = trg_col
        self.max_len = max_len
        self.data[self.src_col] = self.data[self.src_col].astype(str).str.lower().str.strip()
        self.data[self.trg_col] = self.data[self.trg_col].astype(str).str.lower().str.strip()
        self.english = spacy.load("en_core_web_sm")
        self.hindi = spacy.blank("hi")
        self.src2idx, self.idx2src = self.build_vocab(self.data[self.src_col], self.hindi)
        self.trg2idx, self.idx2trg = self.build_vocab(self.data[self.trg_col], self.english)

    def tokenize(self, tokenizer, sentence):
        return [token.text.lower() for token in tokenizer(sentence)]

    def build_vocab(self, texts, tokenizer, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(tokenizer, text)
            counter.update(tokens)
        vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        itos = {i: tok for tok, i in vocab.items()}
        return vocab, itos

    def encode(self, sentence, vocab, tokenizer):
        tokens = self.tokenize(tokenizer, sentence)
        ids = [vocab[SOS_TOKEN]] + [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens] + [vocab[EOS_TOKEN]]
        ids = ids[:self.max_len]
        while len(ids) < self.max_len:
            ids.append(vocab[PAD_TOKEN])
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_sent = self.data.iloc[idx][self.src_col]
        trg_sent = self.data.iloc[idx][self.trg_col]
        src_tokens = self.encode(src_sent, self.src2idx, self.hindi)
        trg_tokens = self.encode(trg_sent, self.trg2idx, self.english)
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(trg_tokens, dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim=64, nhead=4, num_layers=1, max_len=20, trg_pad_idx=0):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=trg_pad_idx)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, emb_dim, padding_idx=trg_pad_idx)
        self.positional_encoding = nn.Parameter(self._create_positional_encoding(max_len, emb_dim), requires_grad=False)

        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.fc_out = nn.Linear(emb_dim, trg_vocab_size)
        self.trg_pad_idx = trg_pad_idx

        self._init_weights()

    def _create_positional_encoding(self, max_len, emb_dim):
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, trg):
        src_emb = self.src_tok_emb(src) * math.sqrt(self.emb_dim) + self.positional_encoding[:, :src.size(1), :]
        trg_emb = self.trg_tok_emb(trg) * math.sqrt(self.emb_dim) + self.positional_encoding[:, :trg.size(1), :]

        src_key_padding_mask = (src == self.trg_pad_idx)
        trg_key_padding_mask = (trg == self.trg_pad_idx)
        trg_mask = torch.tril(torch.ones(trg.size(1), trg.size(1), device=trg.device)).bool()
        output = self.transformer(
            src_emb,
            trg_emb,
            tgt_mask=~trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask
        )
        return self.fc_out(output)

def ids_to_sentence(ids, idx2word):
    return [idx2word[i] for i in ids if i not in (0,1,2)]

def compute_bleu(preds, refs, idx2word):
    pred_sentences = [ids_to_sentence(p, idx2word) for p in preds]
    ref_sentences = [[ids_to_sentence(r, idx2word)] for r in refs]  # Keep corpus_bleu format
    smoothie = SmoothingFunction().method4
    return corpus_bleu(ref_sentences, pred_sentences, smoothing_function=smoothie)

def translate_autoregressive(model, src, src_vocab, trg_vocab, idx2trg, max_len=MAX_LEN):
    model.eval()
    src = src.to(device)
    batch_size = src.size(0)

    # Generate source embeddings and masks once
    src_key_padding_mask = (src == src_vocab[PAD_TOKEN])

    # Prepare src embeddings (with scaling and positional encoding)
    src_emb = model.src_tok_emb(src) * math.sqrt(model.emb_dim)
    src_emb = src_emb + model.positional_encoding[:, :src.size(1), :]

    generated_tokens = torch.full((batch_size, 1), trg_vocab[SOS_TOKEN], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        trg_emb = model.trg_tok_emb(generated_tokens) * math.sqrt(model.emb_dim)
        trg_emb = trg_emb + model.positional_encoding[:, :generated_tokens.size(1), :]

        trg_mask = torch.tril(torch.ones(generated_tokens.size(1), generated_tokens.size(1), device=device)).bool()
        trg_key_padding_mask = (generated_tokens == trg_vocab[PAD_TOKEN])

        output = model.transformer(
            src_emb,
            trg_emb,
            tgt_mask=~trg_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask
        )
        output_logits = model.fc_out(output)  # (batch_size, seq_len, vocab_size)
        next_token_logits = output_logits[:, -1, :]  # last token logits
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)

        # Stop if all sequences have generated EOS_TOKEN
        if (next_tokens == trg_vocab[EOS_TOKEN]).all():
            break

    # Return list of token id lists (exclude SOS and after EOS)
    translations = []
    for seq in generated_tokens.cpu().numpy():
        tokens = []
        for idx in seq[1:]:
            if idx == trg_vocab[EOS_TOKEN]:
                break
            tokens.append(int(idx))
        translations.append(tokens)
    return translations

def main():
    csv_file = "/home/ibab/Desktop/DeepLearning_Lab/Lab16/Hindi_English_Truncated_Corpus.csv"
    max_len = 20
    dataset = TranslationDataset(csv_file, max_len=max_len)
    train_size = int(0.8*len(dataset))
    test_size = len(dataset)-train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8)

    INPUT_DIM = len(dataset.src2idx)
    OUTPUT_DIM = len(dataset.trg2idx)
    model = TransformerSeq2Seq(INPUT_DIM, OUTPUT_DIM, emb_dim=128, max_len=max_len,
                               trg_pad_idx=dataset.trg2idx[PAD_TOKEN]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.trg2idx[PAD_TOKEN])

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]"):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()

            # Forward pass - input target excluding last token for teacher forcing
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]

            # Check for NaN or Inf values in model output
            if not torch.isfinite(output).all():
                print("Skipping batch due to NaN/Inf in model output")
                continue

            # Calculate loss with target shifted by one token
            loss = criterion(output.reshape(-1, output_dim), trg[:, 1:].reshape(-1))

            # Check for NaN or Inf in loss
            if not torch.isfinite(loss):
                print("Skipping batch due to NaN loss")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f}")

        model.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for src, trg in tqdm(test_loader, desc=f"Epoch {epoch + 1} [test]"):
                translations = translate_autoregressive(model, src, dataset.src2idx, dataset.trg2idx, dataset.idx2trg,
                                                        max_len=MAX_LEN)
                all_preds.extend(translations)  # list of list of token ids
                all_refs.extend([r.tolist() for r in trg])
        bleu = compute_bleu(all_preds, all_refs, dataset.idx2trg)
        print(f"Epoch {epoch+1} | BLEU-4: {bleu:.4f}")

    import os
    save_path = "Hin2Eng_Model"
    os.makedirs(save_path, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "seq2seq_model_transformers.pth"))

if __name__ == "__main__":
    main()