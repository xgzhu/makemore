import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# Not used, just fyi
class Head(nn.Module):
    def __init__(self, emb_size, head_size, dropout=0.2):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.WQ = nn.Linear(emb_size, head_size, bias=False)
        self.WK = nn.Linear(emb_size, head_size, bias=False)
        self.WV = nn.Linear(emb_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape (B,T,C)
        B, T, C = x.shape
        Q = self.WQ(x) # (B,T,head_size)
        K = self.WK(x)
        V = self.WV(x)
        weight = Q @ K.transpose(-2,-1) # (B, T, T)
        weight = weight / C ** 0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        weight = self.dropout(weight)
        Z = weight @ V # (B,T,head_size)
        # out = self.WO(Z) # (B, T, C)
        return Z

# Not used, just fyi
class MultiHead(nn.Module):
    def __init__(self, emb_size, head_size, head_cnt):
        super().__init__()
        self.emb_size = emb_size
        self.head_cnt = head_cnt
        self.head_size = head_size
        self.heads = nn.ModuleList(Head(emb_size, head_size) for i in range(head_cnt))
        self.WO = nn.Linear(head_size * head_cnt, emb_size, bias=False)

    def forward(self, x):
        outs = [head(x) for head in self.heads] # [(B, T, C) x head_size]
        print([out.shape for out in outs])
        out = torch.cat(outs, dim=-1)  # (B, T, C x head_size)
        out = self.WO(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, bias = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        op_dim = [-1]
        if not isinstance(self.normalized_shape, int):
            op_dim = [-i for i in range(len(self.normalized_shape), 0, -1)]
        mu = x.mean(dim = op_dim, keepdim = True)
        var = x.var(dim = op_dim, keepdim = True)
        x = (x - mu) * (var + self.eps) ** -0.5 * self.weight
        if self.bias:
            x = x + self.bias
        return x

class MultiHeadFast(nn.Module):
    def __init__(self, emb_size, head_size, head_cnt, max_len):
        super().__init__()
        self.emb_size = emb_size
        self.head_cnt = head_cnt
        self.head_size = head_size
        self.Ws = nn.Linear(emb_size, head_size * head_cnt * 3, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)))
        self.WO = nn.Linear(head_size * head_cnt, emb_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        Q, K, V = torch.tensor_split(self.Ws(x), 3, dim=-1) # (B, T, head_size * head_cnt * 3) -> (B, T, head_size * head_cnt) x3
        Q = Q.view(B, T, self.head_cnt, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.head_cnt, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.head_cnt, self.head_size).transpose(1, 2)
        # Q = self.WQ(x).view(B, T, self.head_cnt, self.head_size).transpose(1, 2) # (B,head_cnt,T,head_size)
        # K = self.WK(x).view(B, T, self.head_cnt, self.head_size).transpose(1, 2)
        # V = self.WV(x).view(B, T, self.head_cnt, self.head_size).transpose(1, 2)

        weight = Q @ K.transpose(-2,-1) # (B,head_cnt,T,T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight * T ** -0.5, dim = 2)
        out = weight @ V # (B,head_cnt,T,head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_size, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, emb_size, max_token, head_size = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.max_token = max_token
        self.head_size = head_size
        self.multi_head_attn = MultiHeadFast(emb_size, head_size, emb_size//head_size, max_token)
        self.ffn = FeedForward(emb_size)
        self.ln1 = LayerNorm(emb_size)
        self.ln2 = LayerNorm(emb_size)

    def forward(self, x):
        x = self.multi_head_attn(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, emb_size, max_token, layer_cnt = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.max_token = max_token
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        self.position_embedding_table = nn.Embedding(max_token, emb_size)
        self.blocks = nn.Sequential(*[TransformerBlock(vocab_size, emb_size, max_token) for _ in range(layer_cnt)])
        self.ln = nn.Linear(emb_size, vocab_size)
        

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        position_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = token_emb + position_emb
        x = self.blocks(x)
        logits = self.ln(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, stop_token = None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
