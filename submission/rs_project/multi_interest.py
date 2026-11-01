import math

import torch

from torch import nn

class MultiInterest(nn.Module):
    def __init__(self,n_items,dim=128,length=100,heads=4,dropout=.2):
        super().__init__();self.items=nn.Embedding(n_items+1,dim,padding_idx=0)
        self.positions=nn.Embedding(length,dim)
        self.attention=nn.Sequential(nn.Linear(dim,dim*4),nn.Tanh(),nn.Linear(dim*4,heads))
        self.norm=nn.LayerNorm(dim,eps=1e-6);self.drop=nn.Dropout(dropout)
        for layer in self.modules():
            if isinstance(layer,(nn.Linear,nn.Embedding)):
                nn.init.normal_(layer.weight,std=.02)
                if isinstance(layer,nn.Linear):nn.init.zeros_(layer.bias)
        with torch.no_grad():self.items.weight[0].zero_()
    def encode(self,tokens):
        mask=tokens>0
        if not mask.any(1).all():raise ValueError('Empty interest history')
        emb=self.drop(self.items(tokens));position=torch.arange(tokens.shape[1],device=tokens.device)
        logits=self.attention(emb+self.positions(position)[None,:,:]).transpose(1,2)
        attention=torch.softmax(logits.masked_fill(~mask[:,None,:],-torch.inf),dim=-1)
        return self.norm(attention@emb)
    def score(self,tokens,method='max',eligible=None):
        logits=self.encode(tokens)@self.items.weight[1:].T
        if method=='max':return logits.max(1).values
        if method=='mixture':
            if eligible is not None:logits=logits.masked_fill(~eligible[:,None,:],-torch.inf)
            score=torch.logsumexp(torch.log_softmax(logits,dim=-1),dim=1)-math.log(logits.shape[1])
            return score if eligible is None else score.masked_fill(~eligible,-1e6)
        raise ValueError('Unknown interest aggregation')

def interest_loss(logits,indices,valid,objective):
    # logits are already masked for observed past items. Every target is unseen
    # in its own observed prefix; only the TRAINING target assigns a readout.
    gather_indices=indices[:,None,:].expand(-1,logits.shape[1],-1)
    if objective=='hard':
        selected=logits.gather(2,gather_indices).argmax(1,keepdim=True)
        logp=torch.log_softmax(logits,dim=2).gather(2,gather_indices).gather(1,selected).squeeze(1)
    elif objective=='mixture':
        target_logp=torch.log_softmax(logits,dim=2).gather(2,gather_indices).masked_fill(~valid[:,None,:],0)
        logp=torch.logsumexp(target_logp,dim=1)-math.log(logits.shape[1])
    else:raise ValueError('Unknown interest objective')
    logp=logp.masked_fill(~valid,0)
    return -(logp.sum(1)/valid.sum(1)).mean()
