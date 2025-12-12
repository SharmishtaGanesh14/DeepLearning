import os
from collections import Counter
import pandas as pd
import spacy
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
        self.english = spacy.load("en_core_web_sm")
        if vocab is None:
            self.word2idx, self.idx2word = self.build_vocab(self.caption_df, self.english)
        else:
            self.word2idx, self.idx2word = vocab
        self.vocab_size = len(self.word2idx)

    def tokenize(self, tokenizer, sentence):
        return [token.text.lower() for token in tokenizer(sentence)]

    def build_vocab(self, captions_df, tokenizer, min_freq=1):
        counter = Counter()
        for caption in captions_df["caption"]:
            tokens = self.tokenize(tokenizer, caption)
            counter.update(tokens)
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for word, count in counter.items():
            if count >= min_freq and word not in vocab:
                vocab[word] = len(vocab)
        itos = {i: w for w, i in vocab.items()}
        return vocab, itos

    def encode_caption(self, caption, max_len=20):
        tokens = self.tokenize(self.english, caption)
        ids = [self.word2idx['<SOS>']] + [self.word2idx.get(w, self.word2idx['<UNK>']) for w in tokens] + [self.word2idx['<EOS>']]
        ids = ids[:max_len]
        ids.extend([self.word2idx['<PAD>']] * (max_len - len(ids)))
        return ids

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

# Encoder CNN
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

# Decoder RNN
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
        h_t = torch.tanh(self.init_h(features)).unsqueeze(0)
        i_t = torch.full((1, batch_size), self.start_idx, dtype=torch.long, device=features.device)
        outputs = []

        for _ in range(max_len):
            embedding = self.embedding(i_t)
            output, h_t = self.rnn(embedding, h_t)
            output = self.fc(output.squeeze(0))
            outputs.append(output)
            _, i_t = torch.max(output, dim=1)
            i_t = i_t.unsqueeze(0)
        return torch.stack(outputs, dim=1)

    def inference(self, features, max_len=20):
        batch_size = features.size(0)
        h_t = torch.tanh(self.init_h(features)).unsqueeze(0)
        i_t = torch.full((1, batch_size), self.start_idx, dtype=torch.long, device=features.device)
        outputs = []

        for _ in range(max_len):
            embedding = self.embedding(i_t)
            output, h_t = self.rnn(embedding, h_t)
            output = self.fc(output.squeeze(0))
            _, i_t = torch.max(output, dim=1)
            outputs.append(i_t.clone())
            i_t = i_t.unsqueeze(0)

            # Early stopping if all sequences predict <EOS>
            if torch.all(i_t == self.end_idx):
                break

        return torch.stack(outputs, dim=1)

# Full model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, start_idx=1, end_idx=2):
        super().__init__()
        self.encoder = EncoderCNN(embedding_size)
        self.decoder = DecoderRNN(hidden_size, embedding_size, vocab_size, start_idx, end_idx)

    def forward(self, images, max_len=20):
        features = self.encoder(images)
        outputs = self.decoder(features, max_len)
        return outputs

    def inference(self, image, max_len=20):
        features = self.encoder(image)
        with torch.no_grad():
            output = self.decoder.inference(features, max_len)
        return output

def ids_to_sentence(ids, idx2word):
    return [idx2word[int(i)] for i in ids if int(i) not in (0, 1, 2)]

def compute_bleu(predictions, references, idx2word):
    pred_sentences = [ids_to_sentence(pred, idx2word) for pred in predictions]
    ref_sentences = [[ids_to_sentence(ref, idx2word)] for ref in references]
    smoothie = SmoothingFunction().method4
    return corpus_bleu(ref_sentences, pred_sentences, smoothing_function=smoothie)

def main():
    image_dir = "/home/ibab/Desktop/DeepLearning_Lab/data/Flick8K/Images"
    caption_file = "/home/ibab/Desktop/DeepLearning_Lab/data/Flick8K/captions.txt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = Flickr8kDataset(image_dir, caption_file, transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8)

    model = ImageCaptioningModel(embedding_size=256,
                                 hidden_size=512,
                                 vocab_size=len(dataset.word2idx)).to(device)

    params = list(model.decoder.parameters()) + list(model.encoder.fc.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.word2idx['<PAD>'])
    optimizer = torch.optim.Adam(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    from tqdm import tqdm

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]"):
            images, captions = images.to(device), captions.to(device)
            outputs = model(images)
            outputs = outputs.view(-1, outputs.shape[-1])
            captions = captions.view(-1)
            loss = criterion(outputs, captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {total_loss / len(train_loader.dataset):.4f}")

        # Validation
        model.eval()
        all_preds, all_refs = [], []
        val_loss = 0
        with torch.no_grad():
            for images, captions in tqdm(test_loader, desc=f"Epoch {epoch + 1} [val]"):
                images, captions = images.to(device), captions.to(device)
                outputs = model(images)
                outputs_flat = outputs.view(-1, outputs.shape[-1])
                captions_flat = captions.view(-1)
                val_loss += criterion(outputs_flat, captions_flat).item() * images.size(0)

                for img, caption in zip(images, captions):
                    pred_ids = model.inference(img.unsqueeze(0), max_len=20)
                    all_preds.append(pred_ids.squeeze(0).tolist())
                    all_refs.append(caption.tolist())

        bleu_score = compute_bleu(all_preds, all_refs, dataset.idx2word)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(test_loader.dataset):.4f} - BLEU: {bleu_score:.4f}")

        # Print a few sample predictions
        for i in range(3):
            pred_sentence = " ".join(ids_to_sentence(all_preds[i], dataset.idx2word))
            ref_sentence = " ".join(ids_to_sentence(all_refs[i], dataset.idx2word))
            print(f"Sample {i+1}:\nPred: {pred_sentence}\nRef : {ref_sentence}\n")

    torch.save(model.state_dict(), "rnn_captioning.pth")

if __name__ == "__main__":
    main()
