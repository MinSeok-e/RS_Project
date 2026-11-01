import numpy as np

import torch

from torch import nn

from .neural_ranker import QueryRanker

class UserQueryRanker(nn.Module):
    def __init__(self,n_features,n_users,dim=8,dropout=.5):
        super().__init__();self.user=nn.Embedding(n_users,dim)
        nn.init.zeros_(self.user.weight);self.drop=nn.Dropout(dropout)
        self.base=QueryRanker(n_features+dim,128,'deepset')
    def forward(self,x,mask,users,items=None):
        u=self.drop(self.user(users))[:,None,:].expand(-1,x.shape[1],-1)
        return self.base(torch.cat((x,u),dim=-1),mask)

def identity_tensors(users,items,groups,maxlen):
    padded=np.zeros((len(groups),maxlen),np.int64);query_users=[];offset=0
    for q,count in enumerate(groups):
        user=users[offset]
        if not (users[offset:offset+count]==user).all():raise AssertionError('Mixed query users')
        query_users.append(user);padded[q,:count]=items[offset:offset+count]+1;offset+=count
    return torch.as_tensor(np.array(query_users),device='cuda'),torch.as_tensor(padded,device='cuda')
