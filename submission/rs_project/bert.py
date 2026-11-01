import numpy as np

import torch

from torch import nn

class MaskedRec(nn.Module):
    def __init__(self,n_items,maxlen,dim=64,dropout=.2):
        super().__init__()
        self.mask_id=n_items+1;self.n_items=n_items
        self.item=nn.Embedding(n_items+2,dim,padding_idx=0)
        self.position=nn.Embedding(maxlen,dim)
        self.norm=nn.LayerNorm(dim);self.drop=nn.Dropout(dropout)
        layer=nn.TransformerEncoderLayer(dim,2,4*dim,dropout,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.LayerNorm(dim))
        self.bias=nn.Parameter(torch.zeros(n_items))
        nn.init.normal_(self.item.weight,std=.02);nn.init.normal_(self.position.weight,std=.02)
        with torch.no_grad():self.item.weight[0].zero_()

    def encode(self,tokens):
        pos=torch.arange(tokens.shape[1],device=tokens.device)
        h=self.drop(self.norm(self.item(tokens)+self.position(pos)[None,:,:]))
        return self.head(self.encoder(h,src_key_padding_mask=tokens==0))

def fit_windows(histories,maxlen):
    seq=[]
    for h in histories:
        for start in range(0,len(h),maxlen//2):
            part=h[start:start+maxlen]
            if len(part)>1:
                a=np.zeros(maxlen,np.int64);a[:len(part)]=part+1;seq.append(a)
    return np.stack(seq)

def mask_inputs(original,mask_id,rate,rng,tail_probability=.5):
    selected=(rng.random(original.shape)<rate)&(original>0)
    for row in range(len(original)):
        length=int((original[row]>0).sum())
        if not selected[row].any():selected[row,rng.integers(length)]=True
        if rng.random()<tail_probability:selected[row,length-1]=True
    inputs=original.copy();inputs[selected]=mask_id
    return inputs,selected

def infer_tokens(histories,maxlen,mask_id,n_masks=1):
    tokens=np.zeros((len(histories),maxlen),np.int64)
    for u,h in enumerate(histories):
        h=h[-(maxlen-n_masks):]
        tokens[u,:len(h)]=h+1;tokens[u,len(h):len(h)+n_masks]=mask_id
    return tokens

def score(model,histories,maxlen,n_masks):
    model.eval();tokens=infer_tokens(histories,maxlen,model.mask_id,n_masks);chunks=[]
    with torch.no_grad():
        for offset in range(0,len(tokens),128):
            t=torch.as_tensor(tokens[offset:offset+128],device='cuda')
            h=model.encode(t);mask=(t==model.mask_id).unsqueeze(-1)
            h=(h*mask).sum(1)/n_masks
            logits=h@model.item.weight[1:model.n_items+1].T+model.bias
            chunks.append(logits.cpu().numpy())
    return np.concatenate(chunks)
