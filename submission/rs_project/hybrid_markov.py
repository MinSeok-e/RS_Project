import numpy as np

import torch

from torch import nn

def examples(histories,context=20,horizon=5):
    users=[];past=[];future=[];starts=[];lengths=[]
    for u,h in enumerate(histories):
        starts.append(len(users));lengths.append(len(h)-1)
        for cut in range(1,len(h)):
            p=np.zeros(context,np.int64);f=np.full(horizon,-1,np.int64)
            tail=np.asarray(h[max(0,cut-context):cut])+1
            p[-len(tail):]=tail
            labels=h[cut:cut+horizon];f[:len(labels)]=labels
            users.append(u);past.append(p);future.append(f)
    return tuple(np.asarray(a) for a in (users,past,future,starts,lengths))

class HybridMarkov(nn.Module):
    def __init__(self,n_users,n_items,dim,personal=True,dropout=.1):
        super().__init__();self.personal=personal
        self.user=nn.Embedding(n_users,dim);self.preference=nn.Embedding(n_items,dim)
        self.source=nn.Embedding(n_items+1,dim,padding_idx=0);self.destination=nn.Embedding(n_items,dim)
        self.bias=nn.Parameter(torch.zeros(n_items));self.drop=nn.Dropout(dropout)
        for layer in [self.user,self.preference,self.source,self.destination]:nn.init.normal_(layer.weight,std=.05)
        with torch.no_grad():self.source.weight[0].zero_()

    def representation(self,users,past):
        mask=past>0
        context=(self.source(past)*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        return self.drop(self.user(users)),self.drop(context)

    def all_scores(self,users,past):
        user,context=self.representation(users,past)
        score=context@self.destination.weight.T+self.bias
        if self.personal:score=score+user@self.preference.weight.T
        return score

    def pair_scores(self,users,past,positive,negative):
        user,context=self.representation(users,past)
        pd=self.destination(positive);nd=self.destination(negative)
        difference=(context*(pd-nd)).sum(1)+self.bias[positive]-self.bias[negative]
        penalty=(context.square().sum()+pd.square().sum()+nd.square().sum())
        if self.personal:
            pp=self.preference(positive);np_=self.preference(negative)
            difference=difference+(user*(pp-np_)).sum(1)
            penalty=penalty+user.square().sum()+pp.square().sum()+np_.square().sum()
        return difference,penalty/(2*len(users))
