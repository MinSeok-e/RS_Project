import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

class QueryRanker(nn.Module):
    def __init__(self,n_features,hidden=64,kind='mlp'):
        super().__init__();self.kind=kind
        self.phi=nn.Sequential(nn.Linear(n_features,hidden),nn.GELU(),nn.Dropout(.1))
        if kind=='set_transformer':
            layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,activation='gelu',batch_first=True,norm_first=True)
            self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.rho=nn.Sequential(nn.Linear(hidden*(3 if kind=='deepset' else 1),hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,x,mask):
        h=self.phi(x)
        if self.kind=='set_transformer':
            h=self.attention(h,src_key_padding_mask=~mask)
        if self.kind=='deepset':
            mean=(h*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
            maximum=h.masked_fill(~mask[...,None],-torch.inf).max(1).values
            h=torch.cat((h,mean[:,None,:].expand_as(h),maximum[:,None,:].expand_as(h)),dim=-1)
        return self.rho(h).squeeze(-1).masked_fill(~mask,-1e4)

def ranking_loss(scores,labels,mask,idcg,objective,query_weights=None):
    if query_weights is not None and objective not in ('list_ce','lambda20'):raise ValueError('Query weights require CE or LambdaRank')
    has_positive=labels.sum(1)>0
    if not has_positive.any():return scores.sum()*0
    if objective=='list_ce':
        loss=-(torch.log_softmax(scores,dim=1)*labels).sum(1)/labels.sum(1).clamp_min(1)
    elif objective=='lambda20':
        with torch.no_grad():
            order=torch.argsort(scores,dim=1,descending=True,stable=True)
            ranks=torch.empty_like(order)
            ranks.scatter_(1,order,torch.arange(scores.shape[1],device=scores.device)[None,:].expand_as(order))
            discount=torch.where(ranks<20,1/torch.log2(ranks+2),0.)
            delta=(discount[:,:,None]-discount[:,None,:]).abs()/idcg[:,None,None]
            pairs=(labels[:,:,None]>0)&(labels[:,None,:]==0)&mask[:,:,None]&mask[:,None,:]
        difference=scores[:,:,None]-scores[:,None,:]
        loss=(F.softplus(-difference)*delta*pairs).sum((1,2))/mask.sum(1)
    elif objective.startswith('approx'):
        from .soft_ranking import approximate_ndcg_loss
        alpha=float(objective.rsplit('_a',1)[1])
        return approximate_ndcg_loss(scores,labels,mask,idcg,alpha,objective.startswith('approx20'))
    elif objective=='list_idcg' or objective.startswith(('ce_lambda_a','list_top')):
        from .ranking_objectives import refined_ranking_loss
        return refined_ranking_loss(scores,labels,mask,idcg,objective)
    else:raise ValueError('Unknown ranking objective')
    if query_weights is not None:
        return (loss[has_positive]*query_weights[has_positive]).mean()
    return loss[has_positive].mean()

def pack(features,labels,groups,maxlen,mean,std):
    x=np.zeros((len(groups),maxlen,features.shape[1]),np.float32)
    y=np.zeros((len(groups),maxlen),np.float32);mask=np.zeros((len(groups),maxlen),bool)
    offset=0
    for q,count in enumerate(groups):
        x[q,:count]=(features[offset:offset+count]-mean)/std
        if labels is not None:y[q,:count]=labels[offset:offset+count]
        mask[q,:count]=True;offset+=count
    if offset!=len(features):raise AssertionError('Invalid ranking groups')
    return [torch.as_tensor(a,device='cuda') for a in (x,y,mask)]
