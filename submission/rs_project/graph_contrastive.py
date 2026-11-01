import torch

from torch import nn

from torch.nn import functional as F

from .neural import GraphFactor

class CrossLayerGraph(GraphFactor):
    def __init__(self,x,dim=64,layers=3,epsilon=.2,contrast_layer=1):
        super().__init__(x,dim,layers)
        if not 1<=contrast_layer<=layers:raise ValueError('Invalid contrast layer')
        self.epsilon=epsilon;self.contrast_layer=contrast_layer
        nn.init.xavier_uniform_(self.user);nn.init.xavier_uniform_(self.item)
    def embeddings(self,perturbed=False):
        user,item=self.user,self.item;users=[];items=[];cross=None
        for layer in range(1,self.layers+1):
            next_user=self.norm@item;next_item=self.norm.T@user
            if perturbed:
                next_user=next_user+self.epsilon*next_user.sign()*F.normalize(torch.rand_like(next_user),dim=1)
                next_item=next_item+self.epsilon*next_item.sign()*F.normalize(torch.rand_like(next_item),dim=1)
            user,item=next_user,next_item;users.append(user);items.append(item)
            if layer==self.contrast_layer:cross=(user,item)
        result=(torch.stack(users).mean(0),torch.stack(items).mean(0))
        return (*result,*cross) if perturbed else result

def identity_contrast(first,second,temperature=.2):
    if first.shape!=second.shape or first.ndim!=2:raise ValueError('Contrast views differ')
    logits=F.normalize(first,dim=1)@F.normalize(second,dim=1).T/temperature
    return F.cross_entropy(logits,torch.arange(len(first),device=first.device))
