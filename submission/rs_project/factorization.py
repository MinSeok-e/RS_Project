import torch

def sample_unseen(x,users):
    neg=torch.randint(x.shape[1],users.shape,device=x.device)
    invalid=x[users,neg]>0
    while invalid.any():
        neg[invalid]=torch.randint(x.shape[1],(int(invalid.sum()),),device=x.device)
        invalid=x[users,neg]>0
    return neg

def als_step(extra,preference,factors,reg):
    n,m=extra.shape;dim=factors.shape[1]
    outer=(factors[:,:,None]*factors[:,None,:]).reshape(m,dim*dim)
    a=(extra@outer).reshape(n,dim,dim)
    a+=factors.T@factors+reg*torch.eye(dim,device=factors.device,dtype=factors.dtype)
    b=(preference+extra)@factors
    chol,info=torch.linalg.cholesky_ex(a)
    if (info!=0).any():raise ValueError('ALS normal equation lost positive definiteness')
    return torch.cholesky_solve(b.unsqueeze(-1),chol).squeeze(-1)
