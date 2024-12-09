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


        
class simple_GRU_cell(nn.Module):
    ''' A simple GRU Cell with Layer Normalization'''
    def __init__(self,embed_dim, hidden_embed_dim, vocab_size):
        super().__init__()
        self.update_gate_linear = nn.Linear(embed_dim + hidden_embed_dim, hidden_embed_dim)
        self.update_layernorm = nn.LayerNorm(normalized_shape=(1, hidden_embed_dim))
        self.reset_gate_linear = nn.Linear(embed_dim + hidden_embed_dim, hidden_embed_dim)
        self.reset_layernorm = nn.LayerNorm(normalized_shape=(1, hidden_embed_dim))
        self.h_new_linear = nn.Linear(embed_dim + hidden_embed_dim, hidden_embed_dim)
        self.h_new_layernorm = nn.LayerNorm(normalized_shape=(1, hidden_embed_dim))
        self.logits_linear = nn.Linear(hidden_embed_dim, vocab_size)
    def forward(self, x, h_prev):
        # x should have shape (B, 1, embed_dim)
        # h_prev should have shape (B, 1, hidden_embed_dim) ---> same as c_prev
        gru_in = torch.cat([h_prev, x], dim=-1) # (B, 1, embed_dim + hidden_embed_dim)
        update_gate = F.sigmoid(self.update_layernorm(self.update_gate_linear(gru_in))) # (B, 1, hidden_embed_dim)
        reset_gate = F.sigmoid(self.reset_layernorm(self.reset_gate_linear(gru_in))) # (B, 1, hidden_embed_dim)
        h_new = F.tanh(self.h_new_layernorm(self.h_new_linear(torch.cat([h_prev * reset_gate, x], dim=-1)))) # (B, 1, hidden_embed_dim)
        h_t = (1 - update_gate)*h_prev + update_gate*h_new # (B, 1, hidden_embed_dim)
        y_t = self.logits_linear(h_t) # (B, 1, vocab_size), output logits.
        return (h_t, y_t)


class gru(nn.Module):
    
    def __init__(self, num_timesteps, embed_dim, hidden_embed_dim, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.s = simple_GRU_cell(embed_dim, hidden_embed_dim, vocab_size)
        self.embedder = embedder(vocab_size = vocab_size, embed_dim = embed_dim)
        
    def forward(self, x, h_init = None):
        # x should have shape (B, num_timesteps)
        # h_init should have shape (B, hidden_embed_dim)
        x = self.embedder(x) # (B, num_timesteps, embed_dim)
        # hidden_embed_dim = hidden dim in this example.
        h = torch.zeros_like(x, device=device) # (B, num_timesteps, embed_dim)
        y = torch.zeros((x.shape[0], x.shape[1], self.vocab_size), device=device) # (B, num_timesteps, vocab_size)
        if (h_init is None):
            h_init = torch.zeros_like(h[:, 0, :], device=device)
        h_tmp = h_init.unsqueeze(1) # (B, 1, hidden_embed_dim = embed_dim)
    
        for i in range(x.shape[1]): # num_timesteps
            x_t = x[:, i, :].unsqueeze(1)
            h_tmp, y_tmp = self.s(x = x_t, h_prev = h_tmp)
            h[:, i, :] = h_tmp.squeeze(1)
            y[:, i, :] = y_tmp.squeeze(1)
        return (h, y)

    def compute_loss(self, x, targets, h_init = None):
        #  `x` should have shape (B, num_timesteps)
        # `h_init` should have shape (B, hidden_embed_dim)
        # `targets` should have shape (B, num_timesteps = T)
        _, yhat = self(x = x, h_init = h_init) # (B, T, vocab_size)
        B, T, vs = yhat.shape
        loss = F.cross_entropy(yhat.view(B*T, vs), targets.view(B*T))
        return loss

    @torch.no_grad()
    def generate(self, x0, h_init = None, max_num_chars = 100):
        B = x0.shape[0]
        # x0 should have shape (B,)
        # h_init should have shape (B, hidden_embed_dim)
        final_res = []
        h0 = h_init
        h0 = h0.unsqueeze(1)
        for i in range(max_num_chars):
            x0 = x0.unsqueeze(-1) # (B, 1)
            x0 = self.embedder(x0) # (B, 1, n_embed)
            h0, logits0 = self.s(x = x0, h_prev = h0)
            prob0 = F.softmax(logits0, dim=-1).squeeze(1) # (B, vocab_size)
            x0 = torch.multinomial(prob0, num_samples=1).squeeze(1) #(B,)
            final_res.append(x0)
        
        final_res = torch.stack(final_res, dim=1).tolist() # (B, len(encoder_list) = T)
        return [decoder(final_res[i]) for i in range(B)]
        

gru_model = gru(num_timesteps = T, embed_dim = embed_dim, hidden_embed_dim=hidden_embed_dim, vocab_size = vocab_size)
gru_model = gru_model.to(device)

torch.manual_seed(1337)
optim = torch.optim.AdamW(gru_model.parameters(), lr=1e-3)
losses = []

@torch.no_grad()
def eval_loss(losses):
    gru_model.eval()
    val_loss = torch.tensor(0.0, device=device)
    train_loss = torch.tensor(0.0, device=device)
    for _ in range(200):
        xv, yv = get_data(B, T, 'val')
        xt, yt = get_data(B, T, 'train')
        h_initt = torch.zeros((B, hidden_embed_dim), device=device)
        h_initv = torch.zeros((B, hidden_embed_dim), device=device)
        train_loss += gru_model.compute_loss(xt, yt, h_initt)
        val_loss += gru_model.compute_loss(xv, yv, h_initv)
    val_loss = val_loss.detach().item()/200.0
    train_loss = train_loss.detach().item()/200.0
    losses.append([train_loss, val_loss])
    gru_model.train()


from tqdm import tqdm

for i in tqdm(range(5_001)):
    optim.zero_grad()
    x, y = get_data(B, T, 'train')
    h_init = torch.zeros((B, hidden_embed_dim), device=device)
    loss = gru_model.compute_loss(x, y, h_init)
    loss.backward()
    optim.step()
    
    if (i % 500 == 0):
        eval_loss(losses)
        print(f"i = {i}, train_loss = {losses[-1][0]}, val_loss = {losses[-1][1]}")

gru_model.eval()

generated_text = gru_model.generate(x0 = torch.zeros((1,), dtype=torch.long, device=device),
                                    h_init = torch.zeros((1,hidden_embed_dim), device=device),
                                    max_num_chars=1_000)

print(generated_text[0])

with open('gru_generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text[0])
losses = np.array(losses)
plt.plot(losses[:,0], marker='x', label='training')
plt.plot(losses[:,1], marker='x', label='validation')
plt.xlabel("training iteration (times 500)")
plt.ylabel("loss, averaged over 200 samples")
plt.legend()
plt.savefig('simple_gru_text_generation_losses.png')