############################################################
# Python: Improve Using The Trick: bigram_v2.py [00:58:28] #
############################################################

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many indepenedent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # NOTE: v2 - number of embedding dimensions. 32 dimensional embeddings
# ----------------

# NOTE: For reproducibility
torch.manual_seed(1337)

# NOTE: Read the data
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# NOTE: Get the encoder and decoder
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars) # NOTE: vocab_size is already declared as a global var here
# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and Test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # first 90% data will be train, rest will be val
train_data = data[:n]
val_data = data[n:]

# NOTE: use data loader to get the batch of inputs and targets
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # Generates a tensor of random integers within the range [0, len(data) - block_size) with shape = (batch_size,). The subtraction is to ensure that (the random index + the block size) does not exceed the bounds of the data.
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # The .to method is used to move the tensor to a specified device (GPU / CPU). in-place change.
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    1. Average loss by 'eval_iters' times to make loss a lot less noisy.
    2. Switch model phases between 'model.eval()' and 'model.train()'.
        Although phase-switching in our current BigramLanguageModel implementation has no effect because we don't have dropout or batch normalization layers, it is a good practice to switch the model phases. 
        This is because it's always good to have in mind that what mode/phase your neural network is currently in because some layers will have different behaviors at training time or inference time.
    3. @torch.no_grad(): context manager tells PyTorch that for everything happened inside the estimate_loss(), we will NOT call .backward() on.
                        i.e., we don't intend to do backprop in this function. i.e., we don't intend to update weights here, simply measuring model's performance 
                        This improves efficiency in memeory use in that PyTorch will not store the intermediate variables in the backward pass.
    """
    out = {}

    model.eval() # Switch the model to evaluation phase/mode

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean() # collect eval_iters=200 losses and average them to get smoothier performance measure

    model.train() # Switch the model back to training phase/mode for continual training

    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self): # NOTE: v2 takes out vocab_size as vocab_size already a global variable.
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # NOTE: v2 switch to n_embd to introduce intermediate phase in order to make the model bigger
        # add positional encoding
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # NOTE: v2 - each position from 0 to block_size-1 will get its own embedding vector of size n_embd
        # lm_head = language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size) # NOTE: v2 go from token embeddings to logits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        # NOTE: v2 - get token embeddings instead
        tok_emb = self.token_embedding_table(idx) # (B, T, C=n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C=n_embd)
        # NOTE: v2 - x not only holds token identities but the positions where these tokens occur
        x = tok_emb + pos_emb # (B, T, C=n_embd)
        # NOTE: v2 - go from token embeddings to get logits
        logits = self.lm_head(x) # (B, T, C=vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# NOTE: Initialize our language model
model = BigramLanguageModel(vocab_size)
m = model.to(device) # move the model parameters (weights & bias) to the device so that all the computations in the training loop happen in GPU. This is in-place change. In other words, m is just another reference to the same object, both m and model are now moved to the same specified device. You can just write it as 'model.to(device)' instead of 'm = model.to(device)'.

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# NOTE: Training loop
for iter in range(max_iters):
    # every once in a while, evaluate the loss on train and val datasets
    # 每训练300轮，监测一下目前model的表现: 先把model切换到evaluation模式，然后计算train和val的loss，最后把model切换回training模式。
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb) # Q: Why use model(xb, yb) instead of m(xb, yb)? A: m is just another reference of the same object. Both m and model are on the same device.
    optimizer.zero_grad(set_to_none=True) # clear previous gradients
    loss.backward() # compute gradients based on current loss
    optimizer.step() # update model's params (weights, bias) based on the gradients

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Make sure to generate the context on the same device as the model
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))