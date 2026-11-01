import torch

from torch import nn

class CrossLayer(nn.Module):
    def __init__(self,dim):
        super().__init__();self.linear=nn.Linear(dim,dim)
        nn.init.normal_(self.linear.weight,std=.01);nn.init.zeros_(self.linear.bias)
    def forward(self,x0,x):return x+x0*self.linear(x)

class CrossQueryRanker(nn.Module):
    def __init__(self,n_features,hidden=128,kind='cross2'):
        super().__init__()
        depth=int(kind.removeprefix('cross'))
        if depth not in (1,2,3):raise ValueError('Unsupported cross depth')
        self.cross=nn.ModuleList([CrossLayer(n_features) for _ in range(depth)])
        self.deep=nn.Sequential(nn.Linear(n_features,hidden),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden,hidden),nn.GELU())
        self.output=nn.Sequential(nn.Linear(n_features+hidden*3,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x,mask):
        x0=x.clamp(-5,5);cross=x0
        for layer in self.cross:cross=layer(x0,cross)
        deep=self.deep(x0)
        mean=(deep*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        maximum=deep.masked_fill(~mask[...,None],-torch.inf).max(1).values
        context=torch.cat((cross,deep,mean[:,None,:].expand_as(deep),maximum[:,None,:].expand_as(deep)),dim=-1)
        return self.output(context).squeeze(-1).masked_fill(~mask,-1e4)
