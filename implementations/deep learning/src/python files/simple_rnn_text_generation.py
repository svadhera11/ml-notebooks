import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.autograd.set_detect_anomaly(True)

# hyperparameters
B = 64
T = 128
embed_dim = 50
hidden_embed_dim = 50 # embed_dim = hidden_embed_dim
device = 'cuda'
# ----

with open('data/tiny_s.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set([c for c in text]))) # this is the vocabulary
vocab_size = len(chars)

c_to_i = {c:i for i, c in enumerate(chars)} # encoding dictionary
i_to_c = {i:c for i, c in enumerate(chars)} # decoding dictionary

# encoder and decoder functions
encoder = lambda string: [c_to_i[s] for s in string]
decoder = lambda int_list: ''.join([i_to_c[i] for i in int_list])

dataset = torch.tensor(encoder(text), device=device)

n = int(0.9*len(dataset)) # split the data into train and test
train_data = dataset[:n]
val_data = dataset[n:]

# get B batches of data, with total length T of each batch

def get_data(B, T, split):
    
    batch = torch.zeros((B, T), dtype=torch.int64, device=device)
    batch_targets = torch.zeros((B, T), dtype=torch.int64, device=device)
    curr_dataset = train_data if split == 'train' else val_data
    start_idx = torch.randint(low=0, high=len(curr_dataset)-T, size=(B,), device=device)
    
    for i in range(B):
        batch[i] = curr_dataset[start_idx[i]:start_idx[i] + T]
        batch_targets[i] = curr_dataset[start_idx[i]+1:start_idx[i] + T+1]

    return batch, batch_targets
    

class embedder(nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim = embed_dim)
        
    def forward(self, x):
        return self.embed_table(x) # (B, T, embed_dim)


        
class simple_rnn_cell(nn.Module):
    
    def __init__(self,embed_dim, hidden_embed_dim, vocab_size):
        super().__init__()
        self.linear_x = nn.Linear(embed_dim, hidden_embed_dim)
        self.linear_h_prev = nn.Linear(hidden_embed_dim, hidden_embed_dim)
        self.linear_h = nn.Linear(hidden_embed_dim, vocab_size)
        
    def forward(self, x, h_prev):
        # x should have shape (B, 1, embed_dim)
        # h_prev should have shape (B, 1, hidden_embed_dim)
        a = self.linear_x(x) + self.linear_h_prev(h_prev) # (B, 1, hidden_embed_dim)
        h = F.tanh(a) # (B,1, hidden_embed_dim)
        o = self.linear_h(h) # (B,1, vocab_size)
        return (h, o)


class rnn(nn.Module):
    
    def __init__(self, num_timesteps, embed_dim, hidden_embed_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.s = simple_rnn_cell(embed_dim, hidden_embed_dim, vocab_size)
        self.embedder = embedder(vocab_size = vocab_size, embed_dim = embed_dim)
        
    def forward(self, x, h_init):
        # x should have shape (B, num_timesteps)
        # h_init should have shape (B, hidden_embed_dim)
        x = self.embedder(x) # (B, num_timesteps, embed_dim)
        h_e = torch.zeros_like(x, device=device) # (B, num_timesteps, embed_dim)
        y_e = torch.zeros((x.shape[0], x.shape[1], self.vocab_size), device=device)
        h_tmp = h_init.unsqueeze(1)
        for i in range(x.shape[1]): # num_timesteps
            h_tmp, y_tmp = self.s.forward(x[:, i, :].unsqueeze(1), h_tmp)
            h_e[:, i, :] = h_tmp.squeeze(1)
            y_e[:, i, :] = y_tmp.squeeze(1)
        return h_e, y_e

    def compute_loss(self, x, targets, h_init):
        # x should have shape (B, num_timesteps)
        # h_init should have shape (B, hidden_embed_dim)
        # y should have shape (B, num_timesteps)
        _, yhat = self.forward(x, h_init) # (B, T, vocab_size)
        B, T, vs = yhat.shape
        loss = F.cross_entropy(yhat.view(B*T, vs), targets.view(B*T))
        return loss

    @torch.no_grad()
    def generate(self, x0, h_init, max_num_chars = 100):
        B = x0.shape[0]
        # x0 should have shape (B,)
        # h_init should have shape (B, hidden_embed_dim)
        final_res = []
        h0 = h_init
        h0 = h0.unsqueeze(1)
        for i in range(max_num_chars):
            x0 = x0.unsqueeze(-1) # (B, 1)
            x0 = self.embedder(x0) # (B, 1, n_embed)
            h0, o0 = self.s.forward(x0, h0)
            prob0 = F.softmax(o0, dim=-1).squeeze(1) # (B, vocab_size)
            x0 = torch.multinomial(prob0, num_samples=1).squeeze(1) #(B,)
            final_res.append(x0)
        
        final_res = torch.stack(final_res, dim=1).tolist() # (B, len(encoder_list) = T)
        return [decoder(final_res[i]) for i in range(B)]
        

rnn_enc = rnn(num_timesteps = T, embed_dim = embed_dim, hidden_embed_dim=hidden_embed_dim, vocab_size = vocab_size)
rnn_enc = rnn_enc.to(device)

torch.manual_seed(1337)
optim = torch.optim.AdamW(rnn_enc.parameters(), lr=1e-3)
losses = []

@torch.no_grad()
def eval_loss(losses):
    rnn_enc.eval()
    val_loss = torch.tensor(0.0, device=device)
    train_loss = torch.tensor(0.0, device=device)
    for _ in range(200):
        xv, yv = get_data(B, T, 'val')
        xt, yt = get_data(B, T, 'train')
        h_initt = torch.zeros((B, hidden_embed_dim), device=device)
        h_initv = torch.zeros((B, hidden_embed_dim), device=device)
        train_loss += rnn_enc.compute_loss(xt, yt, h_initt)
        val_loss += rnn_enc.compute_loss(xv, yv, h_initv)
    val_loss = val_loss.detach().item()/200.0
    train_loss = train_loss.detach().item()/200.0
    losses.append([train_loss, val_loss])
    rnn_enc.train()


from tqdm import tqdm

for i in tqdm(range(5_001)):
    optim.zero_grad()
    x, y = get_data(B, T, 'train')
    h_init = torch.zeros((B, hidden_embed_dim), device=device)
    h, yhat = rnn_enc.forward(x, h_init)
    loss = rnn_enc.compute_loss(x, y, h_init)
    loss.backward()
    optim.step()
    
    if (i % 500 == 0):
        eval_loss(losses)
        print(f"i = {i}, train_loss = {losses[-1][0]}, val_loss = {losses[-1][1]}")

rnn_enc.eval()

generated_text = rnn_enc.generate(x0 = torch.zeros((1,), dtype=torch.long, device=device), h_init = torch.zeros((1,hidden_embed_dim), device=device), max_num_chars=1_000)

print(generated_text[0])
with open('rnn_generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text[0])
losses = np.array(losses)
plt.plot(losses[:,0], marker='x', label='training')
plt.plot(losses[:,1], marker='x', label='validation')
plt.xlabel("training iteration (times 500)")
plt.ylabel("loss, averaged over 200 samples")
plt.legend()
plt.savefig('simple_rnn_text_generation_losses.png')