import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import spacy
import math
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 12
BATCH_SIZE = 64
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1
EPOCHS = 15
LR = 0.0005
MIN_FREQ = 1
NUM_WORKERS = 0

print(f"Using device: {device}")
print(f"Configuration: D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_LAYERS={N_LAYERS}")

# POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# TRANSFORMER MODEL
class TransformerTranslator(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(TransformerTranslator, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.src_pad_idx = 0
        self.tgt_pad_idx = 0
        self._causal_mask_cache = {}

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0, std=self.d_model ** -0.5)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        if tgt_seq_len not in self._causal_mask_cache:
            mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len) * float('-inf'), diagonal=1)
            self._causal_mask_cache[tgt_seq_len] = mask

        tgt_mask = self._causal_mask_cache[tgt_seq_len].to(src.device)
        src_key_padding_mask = (src == self.src_pad_idx).transpose(0, 1)
        tgt_key_padding_mask = (tgt == self.tgt_pad_idx).transpose(0, 1)

        return tgt_mask, src_key_padding_mask, tgt_key_padding_mask

    def forward(self, src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.create_mask(src, tgt)

        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        transformer_out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        transformer_out = self.dropout(transformer_out)
        output = self.fc_out(transformer_out)
        output = output.transpose(0, 1)

        return output


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
def Data_Preprocessing(data, max_len=MAX_LEN, min_freq=MIN_FREQ):
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    # Clean data
    data = data.dropna().reset_index(drop=True)
    data["english_sentence"] = data["english_sentence"].astype(str).str.lower().str.strip()
    data["hindi_sentence"] = data["hindi_sentence"].astype(str).str.lower().str.strip()

    # Filter out empty sentences
    data = data[(data["english_sentence"].str.len() > 0) &
                (data["hindi_sentence"].str.len() > 0)].reset_index(drop=True)

    # Load tokenizers
    try:
        english = spacy.load("en_core_web_sm")
        print("Loaded English tokenizer")
    except OSError:
        print("English tokenizer not found, using blank tokenizer")
        english = spacy.blank("en")

    hindi = spacy.blank("hi")

    def tokenize(tokenizer, sentence):
        return [token.text.lower() for token in tokenizer(sentence) if token.text.strip()]

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

    def encode(sentence, vocab, tokenizer, max_length):
        tokens = tokenize(tokenizer, sentence)
        ids = [vocab[SOS_TOKEN]] + [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens] + [vocab[EOS_TOKEN]]
        ids = ids[:max_length]
        while len(ids) < max_length:
            ids.append(vocab[PAD_TOKEN])
        return ids

    # Build vocabularies
    hindi_texts = data["hindi_sentence"].tolist()
    english_texts = data["english_sentence"].tolist()
    source_vocab, source_itos = build_vocab(hindi_texts, hindi, min_freq)
    target_vocab, target_itos = build_vocab(english_texts, english, min_freq)

    # Encode sequences
    source_sequences, target_sequences = [], []
    for _, row in data.iterrows():
        hi_ids = encode(row["hindi_sentence"], source_vocab, hindi, max_len)
        en_ids = encode(row["english_sentence"], target_vocab, english, max_len)
        source_sequences.append(hi_ids)
        target_sequences.append(en_ids)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        source_sequences, target_sequences, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    Train_dataset = Hin2Eng(X_train, y_train)
    Test_dataset = Hin2Eng(X_test, y_test)
    Train_loader = DataLoader(Train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)
    Test_loader = DataLoader(Test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)

    print(f"Data Ready | Source vocab: {len(source_vocab)}, Target vocab: {len(target_vocab)}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return Train_loader, Test_loader, source_vocab, target_vocab, source_itos, target_itos, hindi, english


# TRAINING FUNCTION
def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    loop = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)

    for batch_idx, (src, tgt) in enumerate(loop):
        src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        try:
            output = model(src, tgt_input)

            output = output.contiguous().view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), avg_loss=total_loss / (batch_idx + 1))

            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    return total_loss / num_batches


# EVALUATION FUNCTION
def evaluate(model, dataloader, criterion, target_itos, device):
    model.eval()
    total_loss = 0
    candidates, references = [], []
    chencherry = SmoothingFunction()

    def normalize_tokens(tokens):
        return [t for t in tokens if t not in ["<pad>", "<sos>", "<eos>", "<unk>"] and t.strip()]

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch_idx, (src, tgt) in enumerate(loop):
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            try:
                output = model(src, tgt_input)

                output_loss = output.contiguous().view(-1, output.shape[-1])
                tgt_output_loss = tgt_output.contiguous().view(-1)
                loss = criterion(output_loss, tgt_output_loss)
                total_loss += loss.item()

                preds = output.argmax(dim=-1)

                for i in range(preds.shape[0]):
                    pred_tokens = []
                    true_tokens = []

                    for idx in preds[i]:
                        token_id = idx.item()
                        if token_id == 2:  # EOS
                            break
                        if token_id in target_itos:
                            pred_tokens.append(target_itos[token_id])

                    for idx in tgt[i][1:]:
                        token_id = idx.item()
                        if token_id == 2:  # EOS
                            break
                        if token_id in target_itos:
                            true_tokens.append(target_itos[token_id])

                    pred_tokens = normalize_tokens(pred_tokens)
                    true_tokens = normalize_tokens(true_tokens)

                    if pred_tokens and true_tokens:
                        candidates.append(pred_tokens)
                        references.append([true_tokens])

                loop.set_postfix(loss=loss.item())

            except RuntimeError as e:
                print(f"Evaluation error in batch {batch_idx}: {e}")
                continue

    if candidates and references:
        try:
            bleu = corpus_bleu(references, candidates,
                               smoothing_function=chencherry.method4,
                               weights=(0.25, 0.25, 0.25, 0.25))
        except ZeroDivisionError:
            bleu = 0.0
    else:
        bleu = 0.0

    return total_loss / len(dataloader), bleu, candidates, references


# TRANSLATION FUNCTION
def translate_sentence(model, sentence, source_vocab, target_vocab,
                       source_tokenizer, target_itos, device, max_len=MAX_LEN):
    model.eval()

    tokens = [tok.text.lower() for tok in source_tokenizer(sentence) if tok.text.strip()]
    src_idx = [source_vocab["<sos>"]] + [source_vocab.get(tok, source_vocab["<unk>"]) for tok in tokens] + [
        source_vocab["<eos>"]]

    if len(src_idx) > max_len:
        src_idx = src_idx[:max_len]
    else:
        src_idx = src_idx + [source_vocab["<pad>"]] * (max_len - len(src_idx))

    src_tensor = torch.tensor(src_idx).unsqueeze(0).to(device)

    with torch.no_grad():
        tgt_indices = [target_vocab["<sos>"]]

        for step in range(max_len - 1):
            current_seq = tgt_indices + [target_vocab["<pad>"]] * (max_len - len(tgt_indices))
            tgt_tensor = torch.tensor(current_seq).unsqueeze(0).to(device)

            try:
                output = model(src_tensor, tgt_tensor)
                last_pos = len(tgt_indices) - 1
                next_token = output[0, last_pos, :].argmax().item()
                tgt_indices.append(next_token)

                if next_token == target_vocab["<eos>"]:
                    break

            except RuntimeError as e:
                print(f"Translation error at step {step}: {e}")
                break

        result_words = []
        for idx in tgt_indices[1:]:
            if idx == target_vocab["<eos>"]:
                break
            if idx in target_itos and target_itos[idx] not in ["<pad>", "<unk>"]:
                result_words.append(target_itos[idx])

        return " ".join(result_words) if result_words else "translation failed"

# MAIN TRAINING LOOP
if __name__ == "__main__":
    print("=" * 70)
    print("HINDI-ENGLISH TRANSLATION USING PyTorch nn.Transformer")
    print("=" * 70)

    # Load data
    try:
        data = pd.read_csv("Hindi_English_Truncated_Corpus.csv")
        print(f"Loaded {len(data)} sentence pairs from file")
    except FileNotFoundError:
        print("Error: Hindi_English_Truncated_Corpus.csv file not found")
        print("Please provide a CSV file with 'hindi_sentence' and 'english_sentence' columns")
        exit(1)

    # Preprocess data
    Train_loader, Test_loader, src_vocab, tgt_vocab, src_itos, tgt_itos, hindi_tok, eng_tok = Data_Preprocessing(data)

    # Initialize model
    model = TransformerTranslator(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        dim_feedforward=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPyTorch Transformer Model")
    print(f"Total Parameters: {total_params:,}")
    print(f"Architecture: {N_LAYERS} layers, {N_HEADS} heads, {D_MODEL} dimensions")

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"], label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)

    # Training loop
    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    best_bleu = 0.0
    patience_counter = 0
    patience = 5

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        # Train
        train_loss = train_epoch(model, Train_loader, criterion, optimizer, device, epoch + 1)

        # Evaluate
        val_loss, bleu_score, translations, references = evaluate(
            model, Test_loader, criterion, tgt_itos, device
        )

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu_score:.4f}")

        # Model saving with early stopping
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_bleu': best_bleu,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'src_itos': src_itos,
                'tgt_itos': tgt_itos
            }, 'best_transformer.pth')
            print(f"Model saved with BLEU: {bleu_score:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"\nTraining Complete!")
    print(f"Best BLEU Score: {best_bleu:.4f}")

    # Load best model for testing
    checkpoint = torch.load('best_transformer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Sample translations
    print("\n" + "=" * 50)
    print("SAMPLE TRANSLATIONS")
    print("=" * 50)

    test_sentences = [
        "मैं खुश हूं",
        "यह एक किताब है",
        "आज मौसम अच्छा है",
        "मुझे खाना पसंद है"
    ]

    for hindi_sent in test_sentences:
        predicted = translate_sentence(
            model, hindi_sent, src_vocab, tgt_vocab, hindi_tok, tgt_itos, device
        )

        print(f"Hindi:     {hindi_sent}")
        print(f"Predicted: {predicted}")
        print("-" * 50)

    print("\nTraining completed successfully!")
