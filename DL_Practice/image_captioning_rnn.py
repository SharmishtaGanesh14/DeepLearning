import os
from collections import Counter
import pandas as pd
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset
class Flickr8kDataset:
    def __init__(self, image_path, caption_file, transform, vocab=None):
        self.image_path = image_path
        self.caption_df = pd.read_csv(caption_file)
        self.transform = transform

        if vocab is None:
            self.vocab = self.create_vocab(self.caption_df, frequency=2)
        else:
            self.vocab = vocab

        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}

    def create_vocab(self, caption_df, frequency=1):
        counter = Counter()
        for caption in caption_df["caption"]:
            counter.update(caption.lower().split())
        vocab = ["<PAD>", "<START>", "<END>", "<UNK>"]
        for w, c in counter.items():
            if c >= frequency:
                vocab.append(w)
        return vocab

    def encode_caption(self, caption, max_len=20):
        words = caption.lower().split()
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in words]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        elif len(tokens) < max_len:
            tokens += [self.word2idx["<PAD>"]] * (max_len - len(tokens))
        return tokens

    def __len__(self):
        return len(self.caption_df)

    def __getitem__(self, idx):
        image_name = self.caption_df.iloc[idx, 0]
        caption = self.caption_df.iloc[idx, 1]
        caption_tokens = self.encode_caption(caption)

        image_path = os.path.join(self.image_path, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(caption_tokens, dtype=torch.long)

# Encoder
class EncoderCNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.network = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(self.resnet.fc.in_features, embedding_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, images):
        features = self.network(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.dropout(features)
        return features

# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, start_idx=1, end_idx=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.init_h = nn.Linear(embedding_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, max_len=20):
        batch_size = features.size(0)
        h_t = torch.tanh(self.init_h(features)).unsqueeze(0)  # (1, batch, hidden)
        i_t = torch.full((1, batch_size), self.start_idx, dtype=torch.long, device=features.device)

        outputs = []
        for _ in range(max_len):
            embedding = self.embedding(i_t)  # (1, batch, embed)
            output, h_t = self.rnn(embedding, h_t)
            output = self.fc(output.squeeze(0))  # (batch, vocab)
            outputs.append(output)

            _, i_t = torch.max(output, dim=1)
            i_t = i_t.unsqueeze(0)

        return torch.stack(outputs, dim=1)  # (batch, max_len, vocab)

    def inference(self, features, max_len=20):
        return self.forward(features, max_len)

# Utility functions
def ids_to_sentence(ids, idx2word):
    return [idx2word[i] for i in ids if i not in (0, 1, 2)]  # ignore <PAD>, <START>, <END>


def compute_bleu(predictions, references, idx2word):
    pred_sentences = [ids_to_sentence(pred, idx2word) for pred in predictions]
    ref_sentences = [[ids_to_sentence(ref, idx2word)] for ref in references]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(ref_sentences, pred_sentences, smoothing_function=smoothie)

# Main training
def main():
    image_path = "/home/ibab/Desktop/DeepLearning_Lab/data/Flick8K/Images"
    captions_file = "/home/ibab/Desktop/DeepLearning_Lab/data/Flick8K/captions.txt"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = Flickr8kDataset(image_path, captions_file, transform)
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    encoder = EncoderCNN(embedding_size=256).to(device)
    decoder = DecoderRNN(hidden_size=512, embedding_size=256, vocab_size=len(dataset.vocab)).to(device)
    params = list(decoder.parameters()) + list(encoder.fc.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx["<PAD>"])
    optimizer = torch.optim.Adam(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 5
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        running_loss = 0

        for images, captions in train_loader:
            images, captions = images.to(device), captions.to(device)
            features = encoder(images)
            outputs = decoder(features)  # no teacher forcing

            # Compute loss (flatten batch and seq)
            outputs = outputs.view(-1, outputs.size(-1))
            captions = captions.view(-1)
            loss = criterion(outputs, captions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        avg_loss = running_loss / len(train_loader.dataset)

        # BLEU evaluation
        encoder.eval()
        decoder.eval()
        all_preds, all_refs = [], []
        val_loss = 0

        with torch.no_grad():
            for images, captions in test_loader:
                images, captions = images.to(device), captions.to(device)
                features = encoder(images)
                outputs = decoder(features)
                outputs_flat = outputs.view(-1, outputs.size(-1))
                captions_flat = captions.view(-1)
                val_loss += criterion(outputs_flat, captions_flat).item() * images.size(0)

                # Generate predictions for BLEU (batch size = 1)
                for f, cap in zip(features, captions):
                    pred_ids = decoder.inference(f.unsqueeze(0), max_len=20)
                    all_preds.append(pred_ids)
                    all_refs.append(cap.tolist())

        val_loss /= len(test_loader.dataset)
        bleu_score = compute_bleu(all_preds, all_refs, dataset.idx2word)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | BLEU-4: {bleu_score:.4f} | Val Loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
