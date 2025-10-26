import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import defaultdict

# Amino acid vocabulary (integer encoding)
AA_VOCAB = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21,  # Unknown
    '<PAD>': 0  # Padding
}

MAX_SEQ_LENGTH = 512  # Maximum sequence length


def load_sequences(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        prot_id = record.id.split('|')[1] if '|' in record.id else record.id
        sequences[prot_id] = str(record.seq)
    return sequences


def load_annotations(terms_file):
    annotations = defaultdict(set)
    df = pd.read_csv(terms_file, sep='\t', header=None, names=['protein_id', 'go_term', 'ontology'])
    for _, row in df.iterrows():
        annotations[row['protein_id']].add(row['go_term'])
    return annotations


def build_go_index(annotations):
    all_terms = set()
    for terms in annotations.values():
        all_terms.update(terms)
    term_to_idx = {term: idx for idx, term in enumerate(sorted(all_terms))}
    return term_to_idx


def encode_sequence(seq, max_len=MAX_SEQ_LENGTH):
    seq = seq[:max_len]
    encoded = [AA_VOCAB.get(aa, AA_VOCAB['X']) for aa in seq]
    padding = [AA_VOCAB['<PAD>']] * (max_len - len(encoded))
    return encoded + padding


def encode_labels(protein_id, annotations, term_to_idx):
    label_vec = np.zeros(len(term_to_idx), dtype=np.float32)
    if protein_id in annotations:
        for term in annotations[protein_id]:
            if term in term_to_idx:
                label_vec[term_to_idx[term]] = 1.0
    return label_vec


class ProteinDataset(Dataset):
    def __init__(self, sequences, annotations, term_to_idx):
        self.protein_ids = list(sequences.keys())
        self.sequences = sequences
        self.annotations = annotations
        self.term_to_idx = term_to_idx

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        prot_id = self.protein_ids[idx]
        seq = self.sequences[prot_id]
        encoded_seq = encode_sequence(seq)
        labels = encode_labels(prot_id, self.annotations, self.term_to_idx)
        return {
            'sequence': torch.LongTensor(encoded_seq),
            'labels': torch.FloatTensor(labels),
            'protein_id': prot_id
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LENGTH, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerProteinPredictor(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, num_classes, max_len=MAX_SEQ_LENGTH,
                 dropout=0.3):
        super(TransformerProteinPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, src):
        # src shape: (batch, seq_length)
        padding_mask = (src == 0)  # True for padding positions

        embedded = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)
        embedded = self.positional_encoding(embedded)

        encoder_output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)

        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoder_output * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)

        out = self.dropout(pooled)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)  # Multi-label classification
        return out


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        sequences = batch['sequence'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequence'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def predict_go_terms(model, sequence, device, term_to_idx, threshold=0.01, max_length=MAX_SEQ_LENGTH):
    model.eval()
    encoded_seq = encode_sequence(sequence, max_length)
    input_tensor = torch.LongTensor([encoded_seq]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)  # Shape: (1, num_classes)
        probs = outputs.cpu().numpy()[0]

    predicted_terms = []
    idx_to_term = {idx: term for term, idx in term_to_idx.items()}
    for idx, prob in enumerate(probs):
        if prob >= threshold:
            predicted_terms.append((idx_to_term[idx], prob))
    predicted_terms.sort(key=lambda x: x[1], reverse=True)
    return predicted_terms


def evaluate_on_test(model, test_fasta, term_to_idx, output_file, device, threshold=0.01, max_predictions=1500):
    test_sequences = load_sequences(test_fasta)

    with open(output_file, 'w') as f_out:
        for prot_id, seq in test_sequences.items():
            preds = predict_go_terms(model, seq, device, term_to_idx, threshold=threshold)
            preds = preds[:max_predictions]

            for go_term, prob in preds:
                f_out.write(f"{prot_id}\t{go_term}\t{prob:.3f}\n")

    print(f"Saved test predictions to {output_file}")


def main():
    train_fasta = '/Users/sharmishtaganesh/Desktop/DeepLearning/Lab17/cafa-6-protein-function-prediction/Train/train_sequences.fasta'
    train_terms = '/Users/sharmishtaganesh/Desktop/DeepLearning/Lab17/cafa-6-protein-function-prediction/Train/train_terms.tsv'
    test_fasta = '/Users/sharmishtaganesh/Desktop/DeepLearning/Lab17/cafa-6-protein-function-prediction/Test/testsuperset.fasta'
    output_file =  '/Users/sharmishtaganesh/Desktop/DeepLearning/Lab17/cafa-6-protein-function-prediction/output_transformers.csv'

    print("Loading sequences and annotations...")
    sequences = load_sequences(train_fasta)
    annotations = load_annotations(train_terms)
    term_to_idx = build_go_index(annotations)
    print(f'Loaded {len(sequences)} sequences with {len(term_to_idx)} unique GO terms.')

    dataset = ProteinDataset(sequences, annotations, term_to_idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = TransformerProteinPredictor(
        vocab_size=len(AA_VOCAB),
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=len(term_to_idx),
        max_len=MAX_SEQ_LENGTH,
        dropout=0.3
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'transformer_protein_function_model.pth')
            print(f'Saved best model at epoch {epoch + 1}.')

    print('Training complete.')

    # Load best saved model for test prediction
    model.load_state_dict(torch.load('transformer_protein_function_model.pth', map_location=device))
    evaluate_on_test(model, test_fasta, term_to_idx, output_file, device)


if __name__ == '__main__':
    main()
