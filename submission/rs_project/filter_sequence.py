import numpy as np

import torch

from torch import nn

def spectral_filter(sequence,weights):
    spectrum=torch.fft.rfft(sequence,dim=1,norm='ortho')
    return torch.fft.irfft(spectrum*torch.view_as_complex(weights),n=sequence.shape[1],dim=1,norm='ortho')

class FilterBlock(nn.Module):
    def __init__(self,length,dim,dropout):
        super().__init__();self.frequency=nn.Parameter(torch.randn(length//2+1,dim,2)*.02)
        self.drop=nn.Dropout(dropout);self.norm1=nn.LayerNorm(dim,eps=1e-6);self.norm2=nn.LayerNorm(dim,eps=1e-6)
        self.feedforward=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,dim))
    def forward(self,x,mask):
        x=self.norm1(x+self.drop(spectral_filter(x,self.frequency)))
        return self.norm2(x+self.drop(self.feedforward(x)))*mask[...,None]

class FilterRec(nn.Module):
    def __init__(self,n_items,length=100,dim=64,dropout=.2,blocks=2):
        super().__init__();self.items=nn.Embedding(n_items+1,dim,padding_idx=0);self.positions=nn.Embedding(length,dim)
        self.norm=nn.LayerNorm(dim,eps=1e-6);self.drop=nn.Dropout(dropout)
        self.blocks=nn.ModuleList([FilterBlock(length,dim,dropout) for _ in range(blocks)])
        for layer in self.modules():
            if isinstance(layer,(nn.Linear,nn.Embedding)):
                nn.init.normal_(layer.weight,std=.02)
                if isinstance(layer,nn.Linear):nn.init.zeros_(layer.bias)
        with torch.no_grad():self.items.weight[0].zero_()
    def encode(self,tokens):
        mask=tokens>0;position=torch.arange(tokens.shape[1],device=tokens.device)
        x=self.drop(self.norm(self.items(tokens)+self.positions(position)[None,:,:]))*mask[...,None]
        for block in self.blocks:x=block(x,mask)
        return x[:,-1]
    def score(self,tokens):return self.encode(tokens)@self.items.weight[1:].T

def prefix_batch(histories,users,ends,length,horizon):
    tokens=np.zeros((len(users),length),np.int64);targets=np.zeros((len(users),horizon),np.int64)
    for row,(u,end) in enumerate(zip(users,ends)):
        h=histories[u]
        if not 1<=end<len(h):raise ValueError('Prefix endpoint outside fitting history')
        prior=h[max(0,end-length):end];future=h[end:end+horizon]
        tokens[row,-len(prior):]=prior+1;targets[row,:len(future)]=future+1
    return tokens,targets

def sample_prefixes(lengths,c,rng):
    if np.any(lengths<2):raise ValueError('Need at least two fitting events')
    if c['sampling']=='users':
        users=np.repeat(np.arange(len(lengths)),c['prefixes_per_user']);rng.shuffle(users)
    else:
        power=c.get('sampling_power',1.)
        mass=(lengths-1) if power==1 else (lengths-1).astype(np.float64)**power
        users=rng.choice(len(lengths),size=len(lengths)*c['prefixes_per_user'],p=mass/mass.sum())
    ratio=c.get('horizon_fraction',0.)
    maxends=np.floor(lengths/(1+ratio)).astype(int) if ratio else lengths-1
    maxends=np.minimum(lengths-1,np.maximum(maxends,1))
    policy=c.get('prefix_policy','uniform')
    if policy not in ('uniform','recent_half'):raise ValueError('Unknown prefix sampling policy')
    low=np.maximum(1,np.ceil(maxends*.5).astype(int)) if policy=='recent_half' else np.ones_like(maxends)
    ends=np.array([rng.integers(low[u],maxends[u]+1) for u in users])
    return users,ends

def configured_prefix_batch(histories,users,ends,c):
    tokens,targets=prefix_batch(histories,users,ends,c['length'],c['horizon'])
    if c.get('horizon_fraction'):
        wanted=np.minimum(c['horizon'],np.maximum(1,np.ceil(ends*c['horizon_fraction']).astype(int)))
        targets[np.arange(c['horizon'])[None,:]>=wanted[:,None]]=0
        if not np.array_equal((targets>0).sum(1),wanted):raise AssertionError('Insufficient future for aligned prefix')
    return tokens,targets

def inference_tokens(histories,length):
    result=np.zeros((len(histories),length),np.int64)
    for u,h in enumerate(histories):
        kept=h[-length:];result[u,-len(kept):]=kept+1
    return result
