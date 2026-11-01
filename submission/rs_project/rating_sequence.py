import math

import numpy as np

import torch

from torch import nn

from .sasrec import SASRec

class RatingSASRec(SASRec):
    def __init__(self,n_items,rating_scale=1.,**kwargs):
        super().__init__(n_items,**kwargs);self.rating_scale=rating_scale
        self.rating_embedding=nn.Embedding(11,self.dim,padding_idx=0)
        nn.init.normal_(self.rating_embedding.weight,std=.02)
        with torch.no_grad():self.rating_embedding.weight[0].zero_()

    def encode(self,tokens,ratings):
        if tokens.shape!=ratings.shape or tokens.ndim!=2 or tokens.shape[1]>self.maxlen:raise ValueError('Bad rated sequence')
        padding=tokens==0
        if padding[:,0].any():raise ValueError('Right-padded nonempty sequence required')
        positions=torch.arange(tokens.shape[1],device=tokens.device)
        ratings=ratings.masked_fill(padding,0)
        x=(self.item_embedding(tokens)+self.rating_scale*self.rating_embedding(ratings))*math.sqrt(self.dim)
        x=self.dropout(x+self.position_embedding(positions)).masked_fill(padding[:,:,None],0)
        causal=torch.ones(tokens.shape[1],tokens.shape[1],device=tokens.device,dtype=torch.bool).triu(1)
        for block in self.blocks:x=block(x,padding,causal)
        return self.final_norm(x).masked_fill(padding[:,:,None],0)

    def score(self,tokens,ratings):
        hidden=self.encode(tokens,ratings);last=(tokens>0).sum(1)-1
        return hidden[torch.arange(len(tokens),device=tokens.device),last]@self.item_embedding.weight[1:].T

def ratings_for_tokens(table,users,tokens):
    return table[users[:,None],(tokens-1).clamp_min(0)].masked_fill(tokens==0,0)

class RatedBatch:
    def __init__(self,model,ratings):self.model,self.ratings=model,ratings
    @property
    def item_embedding(self):return self.model.item_embedding
    def encode(self,tokens):return self.model.encode(tokens,self.ratings)

@torch.no_grad()
def predict_rated(model,tokens,rating_table):
    model.eval();scores=[]
    for start in range(0,len(tokens),128):
        batch=tokens[start:start+128];users=torch.arange(start,start+len(batch),device=tokens.device)
        scores.append(model.score(batch,ratings_for_tokens(rating_table,users,batch)).cpu().numpy())
    return np.concatenate(scores)
