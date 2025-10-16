import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, src, hidden=None):
        # src: [seq_len, batch_size]
        embedded = self.embedding(src)  # [seq_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded, hidden)
        # outputs: [seq_len, batch_size, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        return hidden

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # output_dim = vocab size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, input, hidden):
        # input: [batch_size] -> single token per batch
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.embedding(input)  # [1, batch_size, emb_dim]
        output, hidden = self.rnn(embedded, hidden)  # output: [1, batch_size, hidden_dim]
        prediction = self.fc_out(output.squeeze(0))   # [batch_size, vocab_size]
        return prediction, hidden

# Seq2Seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg=None, max_len=10, teacher_forcing_ratio=0.5):
        # src: [seq_len, batch_size]
        # trg: [seq_len, batch_size] (optional)
        batch_size = src.shape[1]
        trg_len = trg.shape[0] if trg is not None else max_len
        trg_vocab_size = self.decoder.fc_out.out_features

        # Tensor to store predictions
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Initialize hidden state (optional: let encoder do it)
        hidden = self.encoder(src)  # [num_layers, batch_size, hidden_dim]

        # First input to decoder is <sos> token (assume 0)
        input = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        for t in range(trg_len):
            output, hidden = self.decoder(input, hidden)  # output: [batch_size, vocab_size]
            outputs[t] = output

            # Decide if we do teacher forcing
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input = trg[t]  # use ground truth
            else:
                input = output.argmax(1)  # use predicted token

        return outputs  # [seq_len, batch_size, vocab_size]

# Example usage
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VOCAB_SIZE = 10
    EMB_DIM = 8
    HID_DIM = 16
    SEQ_LEN = 5
    BATCH_SIZE = 2

    enc = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM).to(device)
    dec = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    src = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE)).to(device)
    trg = torch.randint(1, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE)).to(device)

    outputs = model(src, trg, max_len=SEQ_LEN, teacher_forcing_ratio=0.7)
    predictions = outputs.argmax(2)  # convert logits -> predicted tokens

    print("Source sequence (input tokens):")
    print(src.T)
    print("\nTarget sequence (true tokens):")
    print(trg.T)
    print("\nPredicted sequence (model output tokens):")
    print(predictions.T)

if __name__ == "__main__":
    main()
