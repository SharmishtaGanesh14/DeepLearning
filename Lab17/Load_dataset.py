import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
from collections import defaultdict

from tqdm import tqdm


# 1. FASTA reader
def read_fasta(fasta_path):
    id_to_seq = {}
    with open(fasta_path) as f:
        entry_id = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if entry_id:
                    id_to_seq[entry_id] = ''.join(seq_lines)
                entry_id = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)
        if entry_id:
            id_to_seq[entry_id] = ''.join(seq_lines)
    return id_to_seq


# 2. TSV reader and vocabulary
def read_tsv(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    id_to_terms = defaultdict(set)
    for _, row in df.iterrows():
        entry_id = row['EntryID']
        go_term = row['term']
        id_to_terms[entry_id].add(go_term)
    # Sorted vocabulary
    all_go_terms = sorted({term for terms in id_to_terms.values() for term in terms})
    return id_to_terms, all_go_terms


# 3. Label vector builder
def make_label_vector(entry_terms, go_term_to_idx):
    label = [0] * len(go_term_to_idx)
    for term in entry_terms:
        if term in go_term_to_idx:
            label[go_term_to_idx[term]] = 1
    return label


# 4. Amino acid mappings
AA = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_idx = {aa: i + 1 for i, aa in enumerate(AA)}  # 0 for pad/unknown


# 5. Custom Dataset
class ProteinFunctionDataset(Dataset):
    def __init__(self, sequences, labels, aa_to_idx, max_length):
        self.sequences = sequences
        self.labels = labels
        self.aa_to_idx = aa_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def encode_sequence(self, seq):
        encoded = [self.aa_to_idx.get(aa, 0) for aa in seq]
        if len(encoded) < self.max_length:
            encoded += [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        seq_encoded = self.encode_sequence(self.sequences[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return seq_encoded, label


# 6. End-to-end assembly from your files
def build_dataset(fasta_path, tsv_path, max_length):
    id_to_seq = read_fasta(fasta_path)
    id_to_terms, go_terms = read_tsv(tsv_path)
    go_term_to_idx = {term: idx for idx, term in enumerate(go_terms)}
    sequences, labels = [], []
    for entry_id in id_to_seq:
        sequences.append(id_to_seq[entry_id])
        entry_terms = id_to_terms.get(entry_id, [])
        label_vec = make_label_vector(entry_terms, go_term_to_idx)
        labels.append(label_vec)
    dataset = ProteinFunctionDataset(
        sequences=sequences,
        labels=labels,
        aa_to_idx=aa_to_idx,
        max_length=max_length
    )
    return dataset, go_terms


# 7. Model Definition
class ProteinFunctionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, seqs):
        emb = self.embedding(seqs)
        out, _ = self.lstm(emb)
        avg_out = out.mean(dim=1)  # mean pooling
        out = self.dropout(avg_out)
        logits = self.fc(out)
        return logits


# 8. Training loop
def train(model, loader, optimizer, loss_fn, device, epochs=5):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for seqs, labels in tqdm(loader,desc=f"Epoch {epoch + 1} [train]"):
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}] Loss: {epoch_loss / len(loader):.4f}")


# 9. Practical usage example
fasta_file = '/home/ibab/Desktop/DeepLearning_Lab/Lab17/cafa-6-protein-function-prediction/Train/train_sequences.fasta'
tsv_file = '/home/ibab/Desktop/DeepLearning_Lab/Lab17/cafa-6-protein-function-prediction/Train/train_terms.tsv'
max_length = 1024

dataset, go_terms = build_dataset(fasta_file, tsv_file, max_length)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
vocab_size = len(aa_to_idx) + 1  # padding idx=0
embedding_dim = 64
hidden_dim = 128
output_dim = len(go_terms)
pad_idx = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ProteinFunctionRNN(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

train(model, loader, optimizer, loss_fn, device, epochs=5)
