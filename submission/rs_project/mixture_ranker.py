import torch

from torch import nn

from .neural_ranker import ranking_loss

class MixtureQueryRanker(nn.Module):
    def __init__(self,n_features,hidden=128,kind='mixture2',auxiliary_weight=0.):
        super().__init__();self.auxiliary_weight=auxiliary_weight;self.use_gate=not kind.startswith('mean');self.floor=.1
        self.n_experts=int(kind.removeprefix('mixture' if self.use_gate else 'mean'))
        if self.n_experts not in (2,4,8,16,32,64):raise ValueError('Unsupported expert count')
        self.phi=nn.Sequential(nn.Linear(n_features,hidden),nn.GELU(),nn.Dropout(.1))
        self.gate=nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,self.n_experts))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(hidden*3,hidden),nn.GELU(),nn.Linear(hidden,1)) for _ in range(self.n_experts)])
    def components(self,x,mask):
        h=self.phi(x)
        mean=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        maximum=h.masked_fill(~mask[...,None],-torch.inf).max(1).values
        context=torch.cat((mean,maximum),dim=-1)
        gates=((1-self.floor)*torch.softmax(self.gate(context),dim=-1)+self.floor/self.n_experts
               if self.use_gate else context.new_full((len(context),self.n_experts),1/self.n_experts))
        features=torch.cat((h,context[:,None,:].expand(-1,h.shape[1],-1)),dim=-1)
        expert_scores=torch.cat([head(features) for head in self.heads],dim=-1)
        return expert_scores,gates
    def forward(self,x,mask):
        scores,gates=self.components(x,mask)
        return (scores*gates[:,None,:]).sum(-1).masked_fill(~mask,-1e4)

def mixture_training_loss(model,x,mask,labels,idcg,objective,query_weights=None):
    scores,gates=model.components(x,mask)
    combined=(scores*gates[:,None,:]).sum(-1).masked_fill(~mask,-1e4)
    loss=ranking_loss(combined,labels,mask,idcg,objective,query_weights)
    if model.auxiliary_weight:
        logp=torch.log_softmax(scores.masked_fill(~mask[...,None],-1e4),dim=1)
        per_query=-(logp*labels[...,None]).sum(1).mean(1)/labels.sum(1).clamp_min(1)
        has_positive=labels.sum(1)>0
        if query_weights is not None:per_query=per_query*query_weights
        if has_positive.any():loss=loss+model.auxiliary_weight*per_query[has_positive].mean()
    return loss
