import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Hyperparameter
batch_size = 16     # how many independent sequences will we process in parallel?
block_size = 32     # what is the maximum context length for predictions?
embedding_dim = 32
hidden_dim = 128
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Tokenizing
with open(project_dir + "/data/word_to_idx.json") as word_to_idx_file:
    word_to_idx = json.load(word_to_idx_file)

with open(project_dir + "/data/idx_to_word.json") as idx_to_word_file:
    idx_to_word = json.load(idx_to_word_file)

vocab_size = len(word_to_idx)

with open("./data/subword_tokenize.txt") as f:
    text = f.read()
words_tokens = text.split(' ')

# Encoding
def encode(words):
    global vocab_size
    result = []
    for word in words:
        try:
            result.append(word_to_idx[word])
        except KeyError:
            label = len(word_to_idx)
            word_to_idx[word] = label
            idx_to_word[str(label)] = word
            result.append(word_to_idx[word])
            vocab_size += 1
    return result


# Decoding
def decode(lst):
    words = []
    for i in lst:
        try:
            word = idx_to_word[str(i)]
            words.append(word)
        except KeyError:
            continue
    return ' '.join(words)

# Self-attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # linear transformation that map (batch_size, input_dim) to (batch_size, output_dim)
        # in this case, input_dim = n_embd and output_dim = n_embd
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # used to calculate the attention score which is dot product between query and key
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x)   # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5   # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)   # (B,T,C)
        out = wei @ v   # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


# Multi-head
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# Feed forward
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# Decoder-only Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, dropout):
        super().__init__()
        # multi-head self-attention
        self.sa = MultiHeadAttention(num_heads, head_size)
        # layer normalization for self-attention
        self.norm1 = nn.LayerNorm(n_embd)
        # dropout regulation to prevent over-fitting for self-attention
        self.dropout1 = nn.Dropout(dropout)
        # feed forwarding
        self.ffwd = FeedForward(n_embd)
        # layer normalization for feed forward
        self.norm2 = nn.LayerNorm(n_embd)
        # dropout for feed forward
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        x = x + self.dropout1(self.sa(self.norm1(x)))
        # Layer normalization and residual connection
        x = x + self.dropout2(self.ffwd(self.norm2(x)))
        return x


# Implement a language model with the decoder block
class NokkaewLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # embedding layer which coverts input token into vector representation
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # embedding layer which generates embeddings for the positions of each token in the sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # layer of the decoders
        self.blocks = nn.Sequential(*[TransformerDecoderBlock(n_embd, n_head, n_embd//n_head, dropout) for _ in range(n_layer)])
        # layer normalization for final layer
        self.ln_f = nn.LayerNorm(n_embd)
        # linear layer which coverts final transformer block into logits for each token in the vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)   # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))     # (T,C)
        x = tok_emb + pos_emb   # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)    # (B,T,C)
        logits = self.lm_head(x)    # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        with torch.no_grad():
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -block_size:]
                # get the predictions
                logits, _ = self(idx_cond)
                # focus only on the last time step
                logits = logits[:, -1, :]   # becomes (B, C)
                # apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)   # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)     # (B, T+1)
        return idx


# Split data to train and validation sets
data = torch.tensor(encode(words_tokens))
m = int(0.8*len(data))
n = int(0.9*len(data))
train_data = data[:m]
val_data = data[m:n]
test_data = data[n:]


# Getting sample as a small batch to train the model
def get_batch(split):
    # Generate a mini batch of X and Y
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_set) - block_size, (batch_size, ))
    x = torch.stack([data_set[i: i+block_size] for i in ix])
    y = torch.stack([data_set[i+1: i+block_size+1] for i in ix])
    return x, y
