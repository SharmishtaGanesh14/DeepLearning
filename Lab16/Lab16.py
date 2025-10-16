import spacy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 12
BATCH_SIZE = 64
EMB_DIM = 128
HIDDEN_DIM = 256
EPOCHS = 30
LR = 0.001
MIN_FREQ = 1

# DATASET CLASS
class Hin2Eng(Dataset):
    def __init__(self, source_list, target_list):
        self.source_tensor = torch.tensor(source_list, dtype=torch.long)
        self.target_tensor = torch.tensor(target_list, dtype=torch.long)

    def __len__(self):
        return len(self.source_tensor)

    def __getitem__(self, idx):
        return self.source_tensor[idx], self.target_tensor[idx]

# DATA PREPROCESSING
def Data_Preprocessing(data):
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    # Clean text
    data["english_sentence"] = data["english_sentence"].astype(str).str.lower().str.strip()
    data["hindi_sentence"] = data["hindi_sentence"].astype(str).str.lower().str.strip()

    # Tokenizers
    english = spacy.load("en_core_web_sm")
    hindi = spacy.blank("hi")

    def tokenize(tokenizer, sentence):
        return [token.text.lower() for token in tokenizer(sentence)]

    # Build vocabulary
    def build_vocab(texts, tokenizer, min_freq=1):
        counter = Counter()
        for text in texts:
            tokens = tokenize(tokenizer, text)
            counter.update(tokens)

        vocab = {PAD_TOKEN: 0, SOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
        for tok, freq in counter.items():
            if freq >= min_freq and tok not in vocab:
                vocab[tok] = len(vocab)
        itos = {i: tok for tok, i in vocab.items()}
        return vocab, itos

    def encode(sentence, vocab, tokenizer, max_length=MAX_LEN):
        tokens = tokenize(tokenizer, sentence)
        ids = [vocab[SOS_TOKEN]] + [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens] + [vocab[EOS_TOKEN]]
        ids = ids[:max_length]
        while len(ids) < max_length:
            ids.append(vocab[PAD_TOKEN])
        return ids

    hindi_texts = data["hindi_sentence"].tolist()
    english_texts = data["english_sentence"].tolist()
    source_vocab, source_itos = build_vocab(hindi_texts, hindi, MIN_FREQ)
    target_vocab, target_itos = build_vocab(english_texts, english, MIN_FREQ)

    # Encode sequences
    source_sequences, target_sequences = [], []
    for _, row in data.iterrows():
        hi_ids = encode(row["hindi_sentence"], source_vocab, hindi)
        en_ids = encode(row["english_sentence"], target_vocab, english)
        source_sequences.append(hi_ids)
        target_sequences.append(en_ids)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(source_sequences, target_sequences, test_size=0.2, random_state=42)

    Train_dataset = Hin2Eng(X_train, y_train)
    Test_dataset = Hin2Eng(X_test, y_test)

    Train_loader = DataLoader(Train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    Test_loader = DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data Ready | Source vocab: {len(source_vocab)}, Target vocab: {len(target_vocab)}")
    return Train_loader, Test_loader, source_vocab, target_vocab, source_itos, target_itos, hindi, english

# MODEL COMPONENTS
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, src):
        src = src.transpose(0, 1)
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=MAX_LEN, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1] if trg is not None else max_len
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)

        input = torch.ones(batch_size, dtype=torch.long).to(self.device)  # <sos>

        for t in range(trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = trg is not None and torch.rand(1).item() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs

# TRAINING & EVAL FUNCTIONS
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, target_itos):
    model.eval()
    total_loss = 0
    candidates, references = [], []
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            loss = criterion(output.view(-1, output.shape[-1]), trg.view(-1)) # op shape becomes [seq_len*batch_size, vocab_size]
            total_loss += loss.item()

            preds = output.argmax(2).transpose(0, 1)
            for i in range(preds.shape[0]):
                pred_tokens = []
                for idx in preds[i]:
                    token_id = idx.item()
                    if token_id == 2:  # <eos>
                        break
                    if token_id != 0:  # ignore <pad>
                        pred_tokens.append(target_itos[token_id])
                true_tokens = []
                for idx in trg[i]:
                    token_id = idx.item()
                    if token_id == 2:  # <eos>
                        break
                    if token_id != 0:  # ignore <pad>
                        true_tokens.append(target_itos[token_id])
                candidates.append(pred_tokens)
                references.append([true_tokens])

    return total_loss / len(dataloader), corpus_bleu(candidates, references), candidates, references

# DECODING (TRANSLATION)
def translate_sentence(model, sentence, source_vocab, target_vocab, source_tokenizer, target_itos):
    model.eval()

    # Tokenize and encode source sentence
    tokens = [tok.text.lower() for tok in source_tokenizer(sentence)]
    src_idx = [source_vocab["<sos>"]] + [source_vocab.get(tok, source_vocab["<unk>"]) for tok in tokens] + [
        source_vocab["<eos>"]]
    src_idx = src_idx[:MAX_LEN] + [source_vocab["<pad>"]] * (MAX_LEN - len(src_idx))
    src_tensor = torch.tensor(src_idx).unsqueeze(0).to(device)  # [1, seq_len]

    with torch.no_grad():
        hidden = model.encoder(src_tensor)

    input_token = torch.tensor([target_vocab["<sos>"]], device=device)
    result_tokens = []

    for _ in range(MAX_LEN):
        output, hidden = model.decoder(input_token, hidden)
        pred_token = output.argmax(1).item()
        if pred_token == target_vocab["<eos>"]:
            break
        result_tokens.append(pred_token)
        input_token = torch.tensor([pred_token], device=device)

    # Convert token IDs to words
    result_words = [target_itos[idx] for idx in result_tokens]
    return " ".join(result_words)


if __name__ == "__main__":
    data = pd.read_csv("Hindi_English_Truncated_Corpus.csv")
    Train_loader, Test_loader, src_vocab, tgt_vocab, src_itos, tgt_itos, hindi_tok, eng_tok = Data_Preprocessing(data)

    encoder = Encoder(len(src_vocab), EMB_DIM, HIDDEN_DIM)
    decoder = Decoder(len(tgt_vocab), EMB_DIM, HIDDEN_DIM)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, Train_loader, criterion, optimizer)
        val_loss, bleu_score, translation,references = evaluate(model, Test_loader, criterion, tgt_itos)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ; BLEU Score: {bleu_score:.4f}")

    # Try translation
    print("\nExample Translation:")
    print(translate_sentence(model, data["hindi_sentence"][0], src_vocab, tgt_vocab, hindi_tok, tgt_itos))


