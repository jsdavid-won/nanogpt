# TODO: Follow better coding practices (e.g., pass in global varaibles)
# fmt: off
import torch
import torch.nn.functional as F
from torch import nn

# Hyperparameters
batch_size = 32
block_size = 8
lr = 1e-2
n_steps = 3000
eval_steps = 200

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
        idx = torch.randint(len(train_data) - block_size, (batch_size,))
        data = train_data
    elif split == "val":
        idx = torch.randint(len(val_data) - block_size, (batch_size,))
        data = val_data
    else:
        raise ValueError("``split`` must be 'train' or 'val'")

    X, y = [], []
    for i in idx:
        X.append(data[i : i + block_size])
        y.append(data[i + 1 : i + 1 + block_size])
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


# Bigram model
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Straight idx to logits
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        if targets is None:
            return self.embedding(idx)

        logits = self.embedding(idx) # (B, L, C) / (batch_size, block_size, vocab_size) in this case

        B, L, C = logits.shape
        # Reshaping to match expected shape of ``F.cross_entropy()``
        loss = F.cross_entropy(logits.reshape((B * L, C)), targets.reshape(B * L))
        return logits, loss

    @torch.no_grad()
    def generate(self, input, n_new_tokens):
        """
        input is a tensor of shape (batch_size, seq_length)
        """
        for _ in range(n_new_tokens):
            logits = self(input) # (B, L, C)
            # Only need the predictions for the next char (idx)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (B, C)
            next_idx = torch.multinomial(probs, 1)
            input = torch.cat([input, next_idx], dim=1)
        
        return input


# Training

model = BigramLM(vocab_size)
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
