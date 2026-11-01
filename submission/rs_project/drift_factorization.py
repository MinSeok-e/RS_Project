import numpy as np

import torch

from torch import nn

class DriftFactor(nn.Module):
    def __init__(self,n_users,n_items,dim):
        super().__init__();self.user=nn.Parameter(torch.randn(n_users,dim)*.1)
        self.drift=nn.Parameter(torch.zeros(n_users,dim));self.item=nn.Parameter(torch.randn(n_items,dim)*.1)
        self.bias=nn.Parameter(torch.zeros(n_items))

    def user_at(self,users,t):return self.user[users]+t[:,None]*self.drift[users]

    def scores(self,t):return (self.user+t*self.drift)@self.item.T+self.bias

def temporal_pairs(histories):
    users=np.repeat(np.arange(len(histories)),[len(h) for h in histories]);items=np.concatenate(histories)
    times=np.concatenate([np.linspace(-1,1,len(h)) for h in histories]).astype(np.float32)
    return users,items,times
