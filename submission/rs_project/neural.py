import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_items, hidden=200, latent=64, dropout=.5, variational=False):
        super().__init__()
        self.variational, self.dropout = variational, dropout
        self.encoder = nn.Sequential(nn.Linear(n_items,hidden), nn.Tanh(),
                                     nn.Linear(hidden,latent*(2 if variational else 1)))
        self.decoder = nn.Sequential(nn.Linear(latent,hidden), nn.Tanh(), nn.Linear(hidden,n_items))
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias, std=.001)

    def forward(self, x):
        h = F.dropout(F.normalize(x,p=2,dim=-1), self.dropout, self.training)
        h = self.encoder(h)
        kl = torch.zeros(len(x),device=x.device)
        if self.variational:
            mu, logvar = h.chunk(2,dim=-1)
            h = mu + torch.randn_like(mu)*torch.exp(.5*logvar) if self.training else mu
            kl = -.5*(1 + logvar - mu.square() - logvar.exp()).sum(1)
        return self.decoder(h), kl

def temporal_examples(histories, n_items, horizon=20, tail=100, half_life=0):
    """Create fixed chronological prefix/future pairs strictly within supplied fit data."""
    inputs, targets = [], []
    for h in histories:
        ends = sorted(set(max(1,min(len(h)-1,int(len(h)*f))) for f in [.2,.4,.6,.8,.9]))
        for end in ends:
            prefix = h[max(0,end-tail):end] if tail else h[:end]
            future = h[end:end+horizon] if horizon else h[end:]
            x,y = np.zeros(n_items,np.float32),np.zeros(n_items,np.float32)
            weights = np.exp2(-np.arange(len(prefix)-1,-1,-1)/half_life) if half_life else 1
            x[prefix] = weights; y[future] = 1
            inputs.append(x); targets.append(y)
    return np.stack(inputs),np.stack(targets)

def infer_features(histories, n_items, tail=100, half_life=0):
    inputs = np.zeros((len(histories),n_items),np.float32)
    for u,h in enumerate(histories):
        h = h[-tail:] if tail else h
        weights = np.exp2(-np.arange(len(h)-1,-1,-1)/half_life) if half_life else 1
        inputs[u,h] = weights
    return inputs

class GraphFactor(nn.Module):
    def __init__(self, x, dim=64, layers=0):
        super().__init__()
        self.layers = layers
        self.user = nn.Parameter(torch.empty(len(x),dim))
        self.item = nn.Parameter(torch.empty(x.shape[1],dim))
        nn.init.normal_(self.user,std=.1); nn.init.normal_(self.item,std=.1)
        norm = x / torch.sqrt(x.sum(1).clamp_min(1)[:,None]*x.sum(0).clamp_min(1)[None,:])
        self.register_buffer('norm',norm)

    def embeddings(self):
        u,i = self.user,self.item
        us,items = [u],[i]
        for _ in range(self.layers):
            u,i = self.norm@i, self.norm.T@u
            us.append(u); items.append(i)
        return torch.stack(us).mean(0), torch.stack(items).mean(0)

    def forward(self):
        u,i = self.embeddings()
        return u@i.T
