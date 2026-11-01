import numpy as np

import torch

from .sasrec import SASRec, inference_array

from .sequence_training import make_windows, full_catalog_loss

from .positive_ce import positive_aware_loss

def model_for(ctx,kind):
    return SASRec(ctx.x.shape[1],maxlen=100,dim=256 if kind=='wide' else 128,blocks=2,heads=2,dropout=.2).cuda()

def fixed_sequence(ctx,kind,smoke=False):
    torch.manual_seed(2020);torch.cuda.manual_seed_all(2020);rng=np.random.default_rng(2020)
    model=model_for(ctx,kind);windows=make_windows(ctx.histories,100,'coverage',5,50)
    active_np=np.flatnonzero(ctx.active);active=torch.as_tensor(active_np+1,device='cuda')
    times=np.full(ctx.x.shape,-1,np.int32)
    for u,h in enumerate(ctx.histories):times[u,h]=np.arange(len(h))
    times=torch.as_tensor(times[:,active_np],device='cuda')
    opt=torch.optim.Adam(model.parameters(),lr=.0003 if kind=='wide' else .001,betas=(.9,.98))
    epochs=2 if smoke else 80 if kind=='wide' else 50
    for _ in range(epochs):
        model.train();order=rng.permutation(len(windows.users))
        for offset in range(0,len(order),128):
            idx=order[offset:offset+128];tokens=torch.as_tensor(windows.tokens[idx],device='cuda');labels=torch.as_tensor(windows.targets[idx],device='cuda')
            if kind=='wide':loss=full_catalog_loss(model,tokens,labels,active)
            else:
                pos=torch.as_tensor(windows.positions[idx],device='cuda');users=torch.as_tensor(windows.users[idx],device='cuda')
                loss=positive_aware_loss(model,tokens,labels,active,times[users],pos,'past')
            if not torch.isfinite(loss):raise ValueError('Nonfinite stronger SASRec loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
    return model

def sequence_views(model,ctx):
    tokens=torch.as_tensor(inference_array(ctx.histories,100),device='cuda')
    model.eval();hidden=[]
    with torch.no_grad():
        for j in range(0,len(tokens),128):hidden.append(model.encode(tokens[j:j+128]))
        hidden=torch.cat(hidden);lengths=(tokens>0).sum(1);positions=torch.arange(tokens.shape[1],device='cuda')[None,:]
        embedding=model.item_embedding.weight[1:]
        views={}
        for tail in [1,5,20]:
            valid=(positions<lengths[:,None])&(positions>=lengths[:,None]-tail)
            query=(hidden*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True)
            views[f'hidden_tail{tail}']=(query@embedding.T).cpu().numpy()
        normalized=torch.nn.functional.normalize(embedding,dim=1)
        for tail in [20,0]:
            query=torch.stack([normalized[torch.as_tensor(h[-tail:] if tail else h,device='cuda')].mean(0) for h in ctx.histories])
            views[f'item_profile_tail{tail}']=(torch.nn.functional.normalize(query,dim=1)@normalized.T).cpu().numpy()
    return views
