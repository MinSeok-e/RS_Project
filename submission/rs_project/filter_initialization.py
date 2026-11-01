import numpy as np

import torch

def collaborative_directions(x,dim,pop_power=.25):
    if not 1<=dim<=min(x.shape):raise ValueError('Invalid collaborative dimension')
    z=np.asarray(x,dtype=np.float64)/np.maximum(x.sum(0),1)**pop_power
    values,vectors=np.linalg.eigh(z@z.T)
    latent=(z.T@vectors[:,-dim:])/np.maximum(values[-dim:],1e-12)**.25
    latent/=np.maximum(np.linalg.norm(latent,axis=1,keepdims=True),1e-12)
    latent[x.sum(0)==0]=0
    if not np.isfinite(latent).all():raise ValueError('Nonfinite initialization')
    return latent.astype(np.float32)

def initialize_filter_items(model,x,c):
    if c.get('initialization')!='svd':return
    if x.shape[1]+1!=len(model.items.weight) or c['dim']!=model.items.embedding_dim:raise ValueError('Initialization dimensions differ')
    norm=float(c['svd_norm'])
    if norm<=0:raise ValueError('Embedding norm must be positive')
    values=collaborative_directions(x,c['dim'],c['svd_pop_power'])*norm
    with torch.no_grad():
        model.items.weight[1:].copy_(torch.as_tensor(values,device=model.items.weight.device,dtype=model.items.weight.dtype))
        model.items.weight[0].zero_()
