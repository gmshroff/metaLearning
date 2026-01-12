import numpy as np
import urllib.request
from autograd import Tensor
from nn import NLLLoss, AdamW, LogSoftmax
from transformer import GPT

# 1. Load Data
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
print(f"Downloading Tiny Shakespeare from {url}...")
with urllib.request.urlopen(url) as f:
    text = f.read().decode('utf-8')

# Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = np.array(encode(text), dtype=np.int32)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Vocab size: {vocab_size}")
print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

# 2. Hyperparameters
batch_size = 32 # Reduced for speed/memory
block_size = 64 # Context length
max_iters = 20000
eval_interval = 20
learning_rate = 1e-3
n_embd = 64
n_head = 4
n_layer = 2

# 3. Model
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = NLLLoss()
log_softmax = LogSoftmax(dim=-1)

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data_split) - block_size, batch_size)
    x = np.stack([data_split[i:i+block_size] for i in ix])
    y = np.stack([data_split[i+1:i+block_size+1] for i in ix])
    return Tensor(x), Tensor(y)

@np.errstate(all='ignore') # Suppress numpy warnings during training
def estimate_loss():
    out = {}
    model.eval = True # Not implemented but good practice placeholder
    for split in ['train', 'val']:
        losses = []
        for _ in range(10):
            X, Y = get_batch(split)
            logits = model(X)
            
            # Reshape for loss
            B, T, C = logits.data.shape
            logits_flat = logits.reshape((B*T, C))
            targets_flat = Y.reshape((B*T,))
            
            # LogSoftmax + NLLLoss
            log_probs = log_softmax(logits_flat)
            loss = criterion(log_probs, targets_flat)
            losses.append(loss.data)
        out[split] = np.mean(losses)
    return out

# 4. Training Loop
print("Starting training...")
for iter in range(max_iters):
    # Evaluation
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample batch
    xb, yb = get_batch('train')

    # Forward pass
    logits = model(xb)
    B, T, C = logits.data.shape
    logits_flat = logits.reshape((B*T, C))
    targets_flat = yb.reshape((B*T,))
    
    log_probs = log_softmax(logits_flat)
    loss = criterion(log_probs, targets_flat)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished.")

# 5. Generation
print("Generating text...")
context = np.zeros((1, 1), dtype=np.int32) # Start with 0 (usually \n)
generated = []
model_input = Tensor(context)

for _ in range(1000):
    # Crop context
    idx_cond = model_input if model_input.data.shape[1] <= block_size else \
               Tensor(model_input.data[:, -block_size:])
    
    logits = model(idx_cond)
    logits = logits[:, -1, :] # Last time step
    
    # Softmax
    probs = np.exp(logits.data) / np.sum(np.exp(logits.data), axis=-1, keepdims=True)
    
    # Sample
    idx_next = np.random.choice(vocab_size, p=probs[0])
    generated.append(idx_next)
    
    # Append
    model_input = Tensor(np.concatenate((model_input.data, [[idx_next]]), axis=1))

print(decode(generated))
