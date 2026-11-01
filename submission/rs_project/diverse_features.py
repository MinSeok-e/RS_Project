import numpy as np

import torch

from .common import weighted_history

from .factorization import als_step

def fixed_als(ctx):
    torch.manual_seed(2020);torch.cuda.manual_seed_all(2020)
    active=np.flatnonzero(ctx.active);x=torch.as_tensor(ctx.x[:,active],device='cuda')
    confidence=weighted_history(ctx,rating_power=2)
    extra=torch.as_tensor(5*confidence[:,active],device='cuda');v=torch.randn(x.shape[1],32,device='cuda')*.01
    with torch.no_grad():
        for _ in range(30):
            u=als_step(extra,x,v,10);v=als_step(extra.T,x.T,u,10)
        a=(u@v.T).cpu().numpy()
    scores=np.zeros_like(ctx.x);scores[:,active]=a
    return scores
