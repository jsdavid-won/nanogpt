# TODO: Follow better coding practices (e.g., pass in global varaibles)
# fmt: off
import torch
import torch.nn.functional as F
from torch import nn

# Hyperparameters
batch_size = 32
seq_len = 8
lr = 1e-3
n_steps = 5000
eval_steps = 500
d_model = 32
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {device}")

torch.manual_seed(1337)

# Load data
with open("input.txt") as file:
    words = file.read()

# Extract vocab from data (character-level tokenization)
vocab = sorted(set(words))
vocab_size = len(vocab)

# For mapping from token to index and vice versa
s_to_i = {s: i for i, s in enumerate(vocab)}
i_to_s = {i: s for i, s in enumerate(vocab)}
encode = lambda seq: [s_to_i[ch] for ch in seq]
decode = lambda idxs: "".join([i_to_s[idx] for idx in idxs])

# Wrap data in tensor
data = torch.tensor(encode(words))

# Train/val split
partition_idx = int(len(data) * 0.9)
train_data, val_data = data[:partition_idx], data[partition_idx:]


def get_batch(split):
    if split == "train":
        idx = torch.randint(len(train_data) - seq_len, (batch_size,))
        data = train_data
    elif split == "val":
        idx = torch.randint(len(val_data) - seq_len, (batch_size,))
        data = val_data
    else:
        raise ValueError("``split`` must be 'train' or 'val'")

    X, y = [], []
    for i in idx:
        X.append(data[i : i + seq_len])
        y.append(data[i + 1 : i + 1 + seq_len])
    X = torch.stack(X)
    y = torch.stack(y)

    return X, y


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_steps)
        for i in range(eval_steps):
            Xb, yb = get_batch(split)
            Xb, yb = Xb.to(device), yb.to(device)
            _, loss = model(Xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    # Normally would just leave this to ``train()`` function, but no such function here
    model.train()
    return out


class SingleheadAttention(nn.Module):
    def __init__(self, d_head=d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.key = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x) # (B, L, d_model)
        scores = (q @ k.transpose(-2, -1)) / d_model**0.5
        # Using truncated tril when masking to accept shorter sequences
        l = x.shape[1]
        scores = scores.masked_fill(self.tril[:l, :l] == 0, -torch.inf)
        return torch.softmax(scores, dim=-1) @ v # (B, L, d_model)

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.heads = nn.ModuleList([SingleheadAttention(d_model // num_heads) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


# Bigram model
class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_enc = nn.Embedding(seq_len, d_model)
        self.attn_head = MultiheadAttention(4)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        x = self.embedding(idx) # (B, L, d_model)
        x += self.positional_enc(torch.arange(x.shape[1], device=device)) # (L, d_model)
        x = self.attn_head(x)
        logits = self.head(x) # (B, L, vocab_size)

        if targets is None:
            return logits, None
        B, L, V = logits.shape
        # Reshaping to match expected shape of ``F.cross_entropy()``
        loss = F.cross_entropy(logits.reshape((B * L, V)), targets.reshape(B * L))
        return logits, loss

    @torch.no_grad()
    def generate(self, input, n_new_tokens):
        """
        input is a tensor of shape (batch_size, seq_length)
        """
        output = input
        for _ in range(n_new_tokens):
            input = output[:, -seq_len:] # (B, L)
            logits, _ = self(input) # (B, L, vocab_size)
            # Only need the predictions for the next char (idx)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (B, vocab_size)
            next_idx = torch.multinomial(probs, 1)
            output = torch.cat([output, next_idx], dim=1) # (B, L)
        
        return output


# Training

model = BigramLM()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

model.train()
for step in range(n_steps):
    Xb, yb = get_batch("train")
    Xb, yb = Xb.to(device), yb.to(device)

    optimizer.zero_grad()
    _, loss = model(Xb, yb)
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        losses = estimate_loss()
        train_loss, val_loss = losses["train"], losses["val"]
        print(f"Step: {step}/{n_steps}, train loss: {train_loss}, val_loss: {val_loss}")

# Generate output
input = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(input, 500)[0].tolist()))
