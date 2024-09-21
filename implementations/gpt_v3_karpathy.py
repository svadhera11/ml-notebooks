import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5_000
eval_interval = 500 
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

torch.manual_seed(1337)

with open('data/tiny_s.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder - takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder - take a list of integers, output a string

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val.
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device) # send x and y to the GPU if available.
    return x, y

@torch.no_grad() # ensures PyTorch does not track gradients for this function.
def estimate_loss():
    out = {}
    model.eval() # set model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set the model back to training mode
    return out


class CombinedMultiHeadAttention(nn.Module):
    ''' combining `Head` and `MultiHeadAttention`, treating `heads` as a new dimension '''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size*num_heads, bias=False)
        self.query = nn.Linear(n_embed, head_size*num_heads, bias=False)
        self.value = nn.Linear(n_embed, head_size*num_heads, bias=False)
        self.attn_linear = nn.Linear(head_size * num_heads, n_embed, bias=False)
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, head_size*num_heads)
        q = self.query(x)
        v = self.value(x)
        k = k.view((B, T, self.num_heads, self.head_size)).transpose(1, 2) # (B, self.num_heads, T, self.head_size)
        q = q.view((B, T, self.num_heads, self.head_size)).transpose(1, 2)
        v = v.view((B, T, self.num_heads, self.head_size)).transpose(1, 2)
        attn = q @ k.transpose(-2, -1) # (B,self.num_heads,T,self.head_size) @ (B,self.num_heads,self.head_size,T) -> (B,self.num_heads,T,T)
        attn = attn * (self.head_size**(-0.5)) # normalize
        attn = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # masking for causal self-attention.
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout1(attn)
        attn = attn @ v # (B,self.num_heads,T,T)@(B,self.num_heads,T,self.head_size) -> (B,self.num_heads,T,self.head_size)
        attn = attn.transpose(1, 2).contiguous().view((B, T, self.num_heads*self.head_size)) # (B, T, num_heads*head_size)
        attn = self.attn_linear(attn) # (B, T, n_embed)
        attn = self.dropout2(attn)
        return attn



class FeedForward(nn.Module):
    ''' a simple linear layer followed by a non-linearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # projection for residual connections.
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''
    
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.sa = CombinedMultiHeadAttention(num_heads=n_head,head_size=head_size)
        self.ffwd = FeedForward(n_embed = n_embed)

    def forward(self, x):
        # residual (skip) connections included.
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x



# super simple bigram model
class DecoderLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(num_embeddings = vocab_size,
                                                  embedding_dim = n_embed)
        self.position_embedding_table = nn.Embedding(num_embeddings = block_size, 
                                                    embedding_dim = n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C_n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C_n_embed)
        x = tok_emb + pos_emb # (B, T, C_n_embed)
        x = self.blocks(x) # (B, T, C_n_embed)
        x = self.ln_f(x) # (B, T, C_n_embed)
        logits = self.lm_head(x) # (B,T,C_vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # get the predictions
            if (idx.shape[1] <= block_size):
                logits, loss = self(idx) # use the full sequence
            else:
                logits, loss = self(idx[:, -block_size:]) # truncate to the last block_size tokens.
                # this truncation is needed because positional embeddings beyond block_size are
                # undefined.
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = DecoderLanguageModel()
model = model.to(device) # send model to GPU

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
from tqdm import tqdm
for iter in tqdm(range(max_iters)):
    # every once in a while, evaluate the loss on the train and val sets
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f"step:{iter}, train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    # cleaar the gradients from previous iteration
    optimizer.zero_grad()
    # backprop
    loss.backward()
    # update parameter values
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
model.eval()
generated_text = decode(model.generate(context, max_new_tokens=10_000)[0].tolist())

with open('gpt_v3_output.txt', 'w', encoding='utf-8') as file:
    file.write(generated_text)

print(generated_text[:100])
