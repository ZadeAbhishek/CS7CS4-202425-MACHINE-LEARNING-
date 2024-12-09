import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter
import random

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 64  # Number of independent sequences processed in parallel
block_size = 128  # Maximum context length for predictions
max_iters = 1000  # Number of iterations for computational efficiency
eval_interval = 100  # Frequency of evaluations
learning_rate = 3e-4  # Learning rate
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Device selection
eval_iters = 200  # Number of iterations for evaluation

# Model Parameters
n_embd = 128  # Embedding size
n_head = 4    # Number of attention heads
n_layer = 4   # Number of transformer layers
dropout = 0.2  # Dropout rate for regularization
torch.manual_seed(1337)

# ----------------------------
# Load Datasets
# ----------------------------
with open('input_childSpeech_trainingSet.txt', 'r', encoding='utf-8') as f_train:
    train_text = f_train.read()

with open('input_shakespeare.txt', 'r', encoding='utf-8') as f_test:
    test_text = f_test.read()

# Combine the texts to ensure the vocab is consistent
combined_text = train_text + test_text
chars = sorted(list(set(combined_text)))
vocab_size = len(chars)
print("vocab_size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(test_text), dtype=torch.long)

# ----------------------------
# Data Loading
# ----------------------------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ----------------------------
# GPT Model Definition
# ----------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=True)
        self.query = nn.Linear(n_embd, head_size, bias=True)
        self.value = nn.Linear(n_embd, head_size, bias=True)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ----------------------------
# Test Loss Calculation
# ----------------------------
print("\nCalculating Test Loss for GPT Model...")
test_loss = estimate_loss()['val']
print(f"Test Loss (GPT Model): {test_loss:.4f}")

# ----------------------------
# Baseline Model Implementation
# ----------------------------
train_words = train_text.split()
word_counts = Counter(train_words)
vocab = list(word_counts.keys())

class BaselineLanguageModel:
    def __init__(self, vocab, word_counts):
        self.vocab = vocab
        self.word_counts = word_counts
        self.total_count = sum(word_counts.values())
        self.probabilities = {word: count / self.total_count for word, count in word_counts.items()}

    def generate(self, max_tokens=20):
        return ' '.join(random.choices(self.vocab, weights=[self.probabilities[word] for word in self.vocab], k=max_tokens))

baseline_model = BaselineLanguageModel(vocab, word_counts)

# Dummy Baseline Loss
baseline_vocab_size = len(vocab)
dummy_loss = -torch.log(torch.tensor(1.0 / baseline_vocab_size)).item()
print(f"Dummy Loss (Baseline Model): {dummy_loss:.4f}")

# ----------------------------
# Comparison
# ----------------------------
print("\nComparison:")
print(f"GPT Model Test Loss: {test_loss:.4f}")
print(f"Baseline Model Loss: {dummy_loss:.4f}")