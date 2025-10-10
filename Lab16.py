import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)

    def forward(self, src): #src [seq_len,batch_size]
        embedded = self.embedding(src) #embedded [seq_len,batch_size,embed_size]
        outputs, hidden = self.rnn(embedded) #op [5,2,16] hidden [1,2,16]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(0) # [seq_len,batch_size]
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=10, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = []

        hidden = self.encoder(src)

        input = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        for t in range(max_len):
            output, hidden = self.decoder(input, hidden)
            top1 = output.argmax(1)
            outputs.append(top1.unsqueeze(0))

            if trg is not None and t < trg.shape[0] and torch.rand(1).item() < teacher_forcing_ratio:
                input = trg[t]
            else:
                input = top1

        outputs = torch.cat(outputs, dim=0)
        return outputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE = 10
    EMB_DIM = 8
    HID_DIM = 16
    SEQ_LEN = 5
    BATCH_SIZE = 2

    enc = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM)
    dec = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM)
    model = Seq2Seq(enc, dec, device).to(device)

    src = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE)).to(device)
    trg = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE)).to(device)

    outputs = model(src, trg, max_len=SEQ_LEN, teacher_forcing_ratio=0.7)

    print("Source sequence (input tokens):")
    print(src.T)
    print("\nTarget sequence (true tokens):")
    print(trg.T)
    print("\nPredicted sequence (model output tokens):")
    print(outputs.T)

if __name__ == "__main__":
    main()