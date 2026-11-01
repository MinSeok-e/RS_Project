import torch

def draw_negatives(x,users,weights=None):
    def draw(n):
        return torch.randint(x.shape[1],(n,),device=x.device) if weights is None else torch.searchsorted(weights,torch.rand(n,device=x.device,dtype=weights.dtype))
    negative=draw(len(users));invalid=x[users,negative]>0
    while invalid.any():
        negative[invalid]=draw(int(invalid.sum()));invalid=x[users,negative]>0
    return negative
