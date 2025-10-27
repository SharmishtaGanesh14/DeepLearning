import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import spacy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
MAX_LEN = 20  # keep same as before

class TranslationDataset(Dataset):
    def __init__(self, csv_file, src_col='hindi_sentence', trg_col='english_sentence', max_len=MAX_LEN):
        self.data = pd.read_csv(csv_file)
        self.src_col = src_col
        self.trg_col = trg_col
        self.max_len = max_len

        # Clean & normalize text
        self.data[self.src_col] = self.data[self.src_col].astype(str).str.lower().str.strip()
        self.data[self.trg_col] = self.data[self.trg_col].astype(str).str.lower().str.strip()

        # Tokenizers
        self.english = spacy.load("en_core_web_sm")
        self.hindi = spacy.blank("hi")

        # Build vocabularies
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

# LSTM Encoder
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden  # (h, c)

# LSTM Decoder
class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def forward(self, src, trg):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features

        hidden = self.encoder(src)

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        input = trg[:, 0]  # <sos>
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = top1
        return outputs

# BLEU Utils
def ids_to_sentence(ids, idx2word):
    return [idx2word[i] for i in ids if i not in (0, 1, 2)]  # pad, sos, eos

def compute_bleu(preds, refs, idx2word):
    pred_sentences = [ids_to_sentence(p, idx2word) for p in preds]
    ref_sentences = [[ids_to_sentence(r, idx2word)] for r in refs]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(ref_sentences, pred_sentences, smoothing_function=smoothie)

# Training Loop
def main():
    csv_file = "/home/ibab/Desktop/DeepLearning_Lab/Lab16/Hindi_English_Truncated_Corpus.csv"
    max_len = 20
    dataset = TranslationDataset(csv_file, max_len=max_len)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    INPUT_DIM = len(dataset.src2idx)
    OUTPUT_DIM = len(dataset.trg2idx)
    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    HID_DIM = 128

    encoder = EncoderLSTM(INPUT_DIM, ENC_EMB_DIM, HID_DIM).to(device)
    decoder = DecoderLSTM(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM).to(device)
    model = Seq2Seq(encoder, decoder, trg_pad_idx=dataset.trg2idx[PAD_TOKEN], device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.trg2idx[PAD_TOKEN])

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            loss = criterion(output[:,1:].reshape(-1, output_dim), trg[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss/len(train_loader):.4f}")

        # BLEU evaluation
        model.eval()
        all_preds, all_refs = [], []
        with torch.no_grad():
            for src, trg in test_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg)
                for i in range(src.size(0)):
                    pred_ids = output[i].argmax(1).tolist()
                    all_preds.append(pred_ids)
                    all_refs.append(trg[i].tolist())
        bleu_score = compute_bleu(all_preds, all_refs, dataset.idx2trg)
        print(f"Epoch {epoch+1} | BLEU-4: {bleu_score:.4f}")

if __name__ == "__main__":
    main()
