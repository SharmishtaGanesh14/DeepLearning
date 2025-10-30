# Author: Robert Guthrie (PyTorch tutorial, explained with comments)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # for reproducibility

# PART 1: Simple embedding lookup example
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # vocabulary size = 2, embedding dimension = 5
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print("Embedding for 'hello':")
print(hello_embed)

# PART 2: N-gram Language Model

CONTEXT_SIZE = 2   # how many words of context
EMBEDDING_DIM = 10 # embedding vector size

# Example text (Shakespeare sonnet)
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# Build training data: list of (context_words, target_word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]

print("\nFirst 3 n-grams (context, target):")
print(ngrams[:3])

# Create vocabulary mapping
vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


# Define NGram Language Model
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Fully connected layers
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # Lookup embeddings for context words
        embeds = self.embeddings(inputs).view((1, -1))  # flatten
        out = F.relu(self.linear1(embeds))              # hidden layer
        out = self.linear2(out)                         # output layer
        log_probs = F.log_softmax(out, dim=1)           # log probabilities
        return log_probs


# Train the model
losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):  # small number of epochs for demo
    total_loss = 0
    for context, target in ngrams:
        # Step 1: prepare context indices
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2: clear old gradients
        model.zero_grad()

        # Step 3: forward pass
        log_probs = model(context_idxs)

        # Step 4: compute loss
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5: backward pass & update
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)
print("\nTraining losses (NGram model):")
print(losses)

# Get embedding of a word, e.g. "beauty"
print("\nLearned embedding for 'beauty':")
print(model.embeddings.weight[word_to_ix["beauty"]])

# PART 3: CBOW (Continuous Bag of Words) Model

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# Create vocabulary
vocab = set(raw_text)
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}

# Prepare CBOW training data
data = []
for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    # context = words before + words after
    context = (
        [raw_text[i - j - 1] for j in range(CONTEXT_SIZE)]
        + [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context, target))

print("\nFirst 5 CBOW training samples (context, target):")
print(data[:5])


# Helper function to convert context words -> tensor of indices
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


# Define CBOW model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * 2 * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        # Look up embeddings of context words
        embeds = self.embeddings(inputs).view(1, -1)    # flatten
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# Train CBOW model
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_ix)

        model.zero_grad()
        log_probs = model(context_vector)

        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)

print("\nTraining losses (CBOW model):")
print(losses)

# Example: print embedding for "process"
print("\nLearned embedding for 'process':")
print(model.embeddings.weight[word_to_ix["process"]])
