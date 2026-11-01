import numpy as np

import torch

from torch import nn

class AttentiveSimilarity(nn.Module):
    def __init__(self,n_items,dim=64,attention=True,beta=.5):
        super().__init__();self.attentive=attention;self.beta=beta
        self.source=nn.Embedding(n_items+1,dim,padding_idx=0)
        self.target=nn.Embedding(n_items,dim);self.bias=nn.Embedding(n_items,1)
        self.attention=nn.Sequential(nn.Linear(dim,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.normal_(self.source.weight,std=.05);nn.init.normal_(self.target.weight,std=.05)
        nn.init.zeros_(self.bias.weight);nn.init.zeros_(self.attention[-1].weight);nn.init.zeros_(self.attention[-1].bias)
        with torch.no_grad():self.source.weight[0].zero_()

    def forward(self,past,candidates,warmup=False):
        # B users, C candidates, L history entries. Exclude target identity, including at fitting.
        mask=(past[:,None,:]>0)&(past[:,None,:]!=candidates[:,:,None]+1)
        interaction=self.source(past)[:,None,:,:]*self.target(candidates)[:,:,None,:]
        similarity=interaction.sum(-1)
        if self.attentive and not warmup:
            logits=self.attention(interaction).squeeze(-1).masked_fill(~mask,-1e4)
            log_normalizer=torch.logsumexp(logits,dim=-1,keepdim=True)
            weights=torch.exp(logits-self.beta*log_normalizer)*mask
        else:weights=mask/mask.sum(-1,keepdim=True).clamp_min(1).float().pow(self.beta)
        return (weights*similarity).sum(-1)+self.bias(candidates).squeeze(-1)

@torch.no_grad()
def full_prediction(model,infer,n_items):
    model.eval();result=np.zeros((len(infer),n_items),np.float32)
    for u in range(0,len(infer),16):
        past=infer[u:u+16]
        for i in range(0,n_items,256):
            candidates=torch.arange(i,min(i+256,n_items),device=infer.device)[None,:].expand(len(past),-1)
            result[u:u+len(past),i:i+len(candidates[0])]=model(past,candidates).cpu().numpy()
    return result
