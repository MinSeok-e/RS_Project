"""PyTorch SASRec: causal self-attention and sampled binary cross entropy.

Kang & McAuley, ICDM 2018, https://arxiv.org/abs/1808.09781
This implementation uses pre-LayerNorm residual blocks and right padding.
All training sequences and negative pools are constructed from fit data only.
"""

import math

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x, padding, causal):
        z = self.norm1(x)
        attn, _ = self.attention(z, z, z, attn_mask=causal,
                                 key_padding_mask=padding, need_weights=False)
        x = x + self.dropout(attn)
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(padding.unsqueeze(-1), 0)

class SASRec(nn.Module):
    def __init__(self, n_items, maxlen=100, dim=64, blocks=2, heads=2, dropout=0.2):
        super().__init__()
        self.maxlen, self.dim = maxlen, dim
        # Padding = 0; internal item index i is represented by token i+1.
        self.item_embedding = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([AttentionBlock(dim, heads, dropout) for _ in range(blocks)])
        self.final_norm = nn.LayerNorm(dim)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.maxlen:
            raise ValueError('Expected batch x length, length <= maxlen')
        padding = tokens == 0
        if padding[:, 0].any():
            raise ValueError('Sequences must be nonempty and right padded')
        position = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.item_embedding(tokens) * math.sqrt(self.dim)
        x = self.dropout(x + self.position_embedding(position))
        x = x.masked_fill(padding.unsqueeze(-1), 0)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool,
                            device=tokens.device).triu(1)
        for block in self.blocks:
            x = block(x, padding, causal)
        return self.final_norm(x).masked_fill(padding.unsqueeze(-1), 0)

    def loss(self, tokens, positive, negative):
        hidden = self.encode(tokens)
        mask = positive > 0
        pos = (hidden * self.item_embedding(positive)).sum(dim=-1)
        neg = (hidden * self.item_embedding(negative)).sum(dim=-1)
        return (F.softplus(-pos[mask]) + F.softplus(neg[mask])).mean()

    def score(self, tokens):
        hidden = self.encode(tokens)
        lengths = (tokens > 0).sum(dim=1)
        last = hidden[torch.arange(len(tokens), device=tokens.device), lengths - 1]
        return last @ self.item_embedding.weight[1:].T

def training_arrays(histories, maxlen):
    tokens = np.zeros((len(histories), maxlen), dtype=np.int64)
    positive = np.zeros_like(tokens)
    for u, history in enumerate(histories):
        if len(history) < 2:
            raise ValueError('At least two fit interactions required for SASRec training')
        source, target = history[:-1][-maxlen:], history[1:][-maxlen:]
        tokens[u, :len(source)] = source + 1
        positive[u, :len(target)] = target + 1
    return tokens, positive

def inference_array(histories, maxlen):
    tokens = np.zeros((len(histories), maxlen), dtype=np.int64)
    for u, history in enumerate(histories):
        seq = history[-maxlen:]
        tokens[u, :len(seq)] = seq + 1
    return tokens

def negative_pools(histories, active_items):
    pools = [np.setdiff1d(active_items, history) + 1 for history in histories]
    if any(len(pool) == 0 for pool in pools):
        raise ValueError('Empty negative sampling pool')
    return pools

def sample_negatives(user_indices, positive, pools, rng):
    negatives = np.zeros_like(positive)
    for row, user in enumerate(user_indices):
        mask = positive[row] > 0
        negatives[row, mask] = rng.choice(pools[user], int(mask.sum()), replace=True)
    return negatives

@torch.no_grad()
def predict(model, tokens, device, batch_size=128):
    model.eval()
    scores = []
    for start in range(0, len(tokens), batch_size):
        batch = torch.as_tensor(tokens[start:start + batch_size], device=device)
        scores.append(model.score(batch).cpu().numpy())
    return np.concatenate(scores)
