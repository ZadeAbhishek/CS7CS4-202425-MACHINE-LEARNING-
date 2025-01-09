import os
import re
import math
import random
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.nn import functional as F

BATCH_SIZE     = 16        # higher batch size to better utilize GPU resources
BLOCK_SIZE     = 256       # max context length for GPT
MAX_ITERS      = 3000      # number of training iterations
EVAL_INTERVAL  = 300       # evaluate every 'eval_interval' steps
LEARNING_RATE  = 5e-5      # recommended lower LR for deeper/wider models
DEVICE         = 'mps' if torch.backends.mps.is_available() else 'cpu'
EVAL_ITERS     = 300

# GPT Architecture
N_EMBD         = 512       # embedding dimension
N_HEAD         = 8         # number of attention heads
N_LAYER        = 8         # number of transformer blocks
DROPOUT        = 0.2       # moderate dropout for regularization

torch.manual_seed(1337)  # reproducibility

print("Device Type:", DEVICE)

# Load the augmented melody dataset
with open('inputMelodiesAugmented.txt', 'r', encoding='utf-8') as file_in:
    raw_text = file_in.read()

# Build the character vocabulary
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

print('Characters in Vocab:')
print(''.join(chars))
print(f"Vocab Size: {vocab_size}")

# Map characters <-> integers
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

def encode_text(text_str: str) -> list:
    """Convert a string into a list of integer IDs."""
    return [char_to_int[ch] for ch in text_str]

def decode_integers(int_list: list) -> str:
    """Convert a list of integer IDs back into a string."""
    return ''.join([int_to_char[i] for i in int_list])

# Create train/val splits
data_as_ints = torch.tensor(encode_text(raw_text), dtype=torch.long)
split_index  = int(0.9 * len(data_as_ints))  # first 90% = train, last 10% = val
train_data   = data_as_ints[:split_index]
val_data     = data_as_ints[split_index:]

def get_batch(split: str):
    """
    Return a batch of data (x, y) where x is the input tokens,
    and y is the next-token targets.
    Each sample in the batch is a random slice of length BLOCK_SIZE.
    """
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))
    x_batch = torch.stack([dataset[i:i+BLOCK_SIZE] for i in ix])
    y_batch = torch.stack([dataset[i+1:i+BLOCK_SIZE+1] for i in ix])
    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
    return x_batch, y_batch

@torch.no_grad()
def estimate_loss():
    """
    Evaluate the model on train and val. Return a dict
    with average losses, e.g. {'train': ..., 'val': ...}.
    """
    results = {}
    gpt_model.eval()
    for split_type in ['train', 'val']:
        losses_holder = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split_type)
            _, batch_loss = gpt_model(X, Y)
            losses_holder[k] = batch_loss.item()
        results[split_type] = losses_holder.mean()
    gpt_model.train()
    return results


class SingleHead(nn.Module):
    """
    One head of self-attention. Projects input x into query,key,value,
    then does masked attention with a triangular matrix to enforce
    causal decoding. 
    """

    def __init__(self, head_size: int):
        super().__init__()
        self.key   = nn.Linear(N_EMBD, head_size, bias=True)
        self.query = nn.Linear(N_EMBD, head_size, bias=True)
        self.value = nn.Linear(N_EMBD, head_size, bias=True)

        self.register_buffer('mask_triangle', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout_layer = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k_out = self.key(x)         # (B,T,head_size)
        q_out = self.query(x)       # (B,T,head_size)

        # attention logits
        weights = q_out @ k_out.transpose(-2, -1) * (1.0 / (k_out.shape[-1] ** 0.5))  
        # masked fill
        weights = weights.masked_fill(self.mask_triangle[:T, :T] == 0, float('-inf'))
        # softmax
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout_layer(weights)

        # Weighted sum
        v_out = self.value(x)       # (B,T,head_size)
        out = weights @ v_out       # (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention: run SingleHead multiple times in parallel,
    then combine.
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads_list = nn.ModuleList([SingleHead(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout_layer = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate the outputs from each head
        combined = torch.cat([h(x) for h in self.heads_list], dim=-1)
        # final linear
        combined = self.projection(combined)
        out = self.dropout_layer(combined)
        return out

class FeedForward(nn.Module):
    """
    A 2-layer MLP that expands to 4*N_EMBD and then returns to N_EMBD.
    """

    def __init__(self, emb_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single Transformer block:
      (1) LayerNorm + multi-head attention + residual
      (2) LayerNorm + feedforward + residual
    """

    def __init__(self, emb_size: int, num_heads: int):
        super().__init__()
        # dimension per head
        head_dim = emb_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_dim)
        self.feed_forward   = FeedForward(emb_size)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class SimpleGPTLanguageModel(nn.Module):
    """
    Simple GPT-style language model for next-character prediction.
    """

    def __init__(self):
        super().__init__()
        # token + position embeddings
        self.token_embed_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embed_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

        # stack of Transformer blocks
        self.blocks_stack = nn.Sequential(
            *[TransformerBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        )

        # final normalization & linear head
        self.final_ln = nn.LayerNorm(N_EMBD)
        self.lm_head  = nn.Linear(N_EMBD, vocab_size)

        self.apply(self._init_parameters)

    def _init_parameters(self, module):
        """
        Parameter init: normal(0, 0.02) for nn.Linear, nn.Embedding
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor=None):
        B, T = idx.shape

        # gather embeddings
        token_emb   = self.token_embed_table(idx)  
        position_ids= torch.arange(T, device=DEVICE)
        pos_emb     = self.position_embed_table(position_ids)
        x = token_emb + pos_emb  # shape (B,T,N_EMBD)

        x = self.blocks_stack(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss_val = None
        if targets is not None:
            B,T,C = logits.shape
            logits_2d = logits.view(B*T, C)
            targets_1d= targets.view(B*T)
            loss_val  = F.cross_entropy(logits_2d, targets_1d)

        return logits, loss_val

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            # select last position
            logits_last = logits[:, -1, :]
            probs       = F.softmax(logits_last, dim=-1)
            idx_next    = torch.multinomial(probs, num_samples=1)
            idx         = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_with_perplexity(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Generate tokens while collecting log-probs -> perplexity.
        """
        log_probs_collector = []

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            final_logits = logits[:, -1, :] 
            final_probs  = F.softmax(final_logits, dim=-1)
            log_p        = torch.log(final_probs)
            log_probs_collector.append(log_p)

            chosen = torch.multinomial(final_probs, num_samples=1)
            idx = torch.cat((idx, chosen), dim=1)

        stacked_logs = torch.stack(log_probs_collector, dim=1)  
        # gather the log-probs for chosen tokens
        # idx[:, 1:] => newly generated
        stacked_logs = stacked_logs.gather(2, idx[:, 1:].unsqueeze(2)).squeeze(2)
        avg_log_prob = stacked_logs.mean()
        perplexity   = torch.exp(-avg_log_prob)
        return idx, perplexity

# create model
gpt_model = SimpleGPTLanguageModel()
gpt_model.to(DEVICE)

# Count parameters
params_count = sum(p.numel() for p in gpt_model.parameters()) / 1e6
print(f"Model Size: {params_count:.2f}M parameters")

# Optimizer
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=LEARNING_RATE)

# Training loop
for iteration in range(MAX_ITERS):
    # Evaluate on train & val periodically
    if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
        metrics = estimate_loss()
        print(f"Step {iteration}: train loss {metrics['train']:.4f}, val loss {metrics['val']:.4f}")

    # get batch
    x_in, y_in = get_batch('train')
    # forward
    _, loss_curr = gpt_model(x_in, y_in)
    optimizer.zero_grad(set_to_none=True)
    loss_curr.backward()
    optimizer.step()

# Save final model
MODEL_PATH = "gpt_model_refracted.pth"
torch.save(gpt_model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# For demonstration: Load it back, generate sample text
loaded_model = SimpleGPTLanguageModel()
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.to(DEVICE)
loaded_model.eval()

# Example generation
seed_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # single zero token
gen_output  = loaded_model.generate(seed_context, max_new_tokens=400)
decoded_str = decode_integers(gen_output[0].tolist())
print("Sample Generation:")
print(decoded_str)

output_filename = "model_generated.txt"
with open(output_filename, "w", encoding="utf-8") as f_out:
    f_out.write(decoded_str)
print(f"Model-generated text has been saved to {output_filename}")

def tokenize_line_of_melody(melody_line: str) -> list:
    """
    Splits a single melody line by '|' to obtain tokens.
    Example:
      melody_line: "B3q|Re|C#4q|Rt|E4q"
      returns:     ["B3q", "Re", "C#4q", "Rt", "E4q"]
    """
    parts = [p.strip() for p in melody_line.strip().split('|') if p.strip()]
    return parts

def build_trigram_counts(tokenized_melody_list: list) -> dict:
    """
    Build raw trigram counts from a list-of-lists structure:
      Each sub-list is tokens from one melody line.
    
    returns a dictionary: 
      { (tokA,tokB) : Counter({tokC:count, ...}), ... }
    """
    tri_counts = defaultdict(Counter)
    for tokens in tokenized_melody_list:
        for i in range(len(tokens) - 2):
            context_2 = (tokens[i], tokens[i+1])
            next_tk   = tokens[i+2]
            tri_counts[context_2][next_tk] += 1
    return tri_counts

def normalize_trigram_counts(tri_counts: dict) -> dict:
    """
    Convert trigram counts -> trigram probabilities.
    e.g., 
      tri_counts[(tokA,tokB)] might be { tokC : 7, tokD : 3 }
      => total=10 => tokC=0.7, tokD=0.3
    """
    model_dict = {}
    for context_2, the_counter in tri_counts.items():
        total_ct = sum(the_counter.values())
        model_dict[context_2] = {k: (v/total_ct) for k, v in the_counter.items()}
    return model_dict

def trigram_generate_sequence(trigram_prob_model: dict, start_2tokens: tuple, max_len=50) -> list:
    """
    Generate tokens from a trigram probability model, starting with a 2-token context.
    If context is missing from the model, generation stops.
    """
    new_sequence = list(start_2tokens)
    for _ in range(max_len - 2):
        last_context = (new_sequence[-2], new_sequence[-1])
        if last_context not in trigram_prob_model:
            break
        possible_next = trigram_prob_model[last_context]
        next_tokens   = list(possible_next.keys())
        next_probs    = list(possible_next.values())
        chosen_token  = random.choices(next_tokens, next_probs)[0]
        new_sequence.append(chosen_token)
    return new_sequence

def compute_trigram_perplexity(token_list: list, trigram_model: dict) -> float:
    """
    Compute perplexity of a token list using a trigram probability model.
    If a context or next token is unseen, assign a small fallback probability.
    """
    if len(token_list) < 3:
        return float('inf')

    log2_sum = 0.0
    count    = 0

    for i in range(len(token_list) - 2):
        context_2 = (token_list[i], token_list[i+1])
        next_tk   = token_list[i+2]
        if context_2 in trigram_model and next_tk in trigram_model[context_2]:
            prob_val = trigram_model[context_2][next_tk]
        else:
            prob_val = 1e-10
        
        log2_sum += math.log2(prob_val)
        count    += 1
    
    if count == 0:
        return float('inf')
    avg_log2 = log2_sum / count
    perplexity = 2 ** (-avg_log2)
    return perplexity

def evaluate_with_trigram_demo():
    """
    Demonstrates how to evaluate perplexities for:
    1) A newly generated line from the trigram model,
    2) A line from the GPT-based model (decoded_str).

    Compares them to see which has a lower perplexity under the trigram model.
    """
    # 1) read lines & tokenize
    input_file = "inputMelodiesAugmented.txt"
    with open(input_file, 'r', encoding='utf-8') as f_in:
        melody_lines = [ln.strip() for ln in f_in if ln.strip()]

    tokenized_list = []
    for line in melody_lines:
        tokens_line = tokenize_line_of_melody(line)
        tokenized_list.append(tokens_line)

    # 2) build & normalize trigram counts
    trigram_cts    = build_trigram_counts(tokenized_list)
    trigram_probs  = normalize_trigram_counts(trigram_cts)

    # 3) generate from trigram
    chosen_start_context = ("B3q", "Re")  # sample context that hopefully is in data
    trigram_output = trigram_generate_sequence(trigram_probs, chosen_start_context, max_len=60)
    trigram_line   = "|".join(trigram_output)
    print("Generated line from trigram model:")
    print(trigram_line)
    
    output_filename = "trigram_model_generated.txt"
    with open(output_filename, "w", encoding="utf-8") as f_out:
        f_out.write(trigram_line)

    print(f"trigram model generated text has been saved to {output_filename}")

    # 4) compute perplexity for that line
    px_trigram_line = compute_trigram_perplexity(trigram_output, trigram_probs)
    print("Perplexity of generated line:", px_trigram_line)

    # 5) compare with GPT model's generation => 'decoded_str'
    # decode_str was produced above
    gpt_tokens = tokenize_line_of_melody(decoded_str)
    px_gpt_line= compute_trigram_perplexity(gpt_tokens, trigram_probs)
    print("Model-generated cod perplexity:", px_gpt_line)

    if px_trigram_line < px_gpt_line:
        print("Trigram's own generation has a lower perplexity (fits model better).")
    else:
        print("The external GPT model has a lower/equal perplexity, or they are similar.")

# Finally, call the evaluation if you want to compare perplexities
evaluate_with_trigram_demo()