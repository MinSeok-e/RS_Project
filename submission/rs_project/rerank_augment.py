import numpy as np

from .common import weighted_history

from .linear import user_knn

def augment(ctx,users,items):
    features=[];names=[]
    x=ctx.x
    ratings=np.zeros_like(x)
    for u,h in enumerate(ctx.histories):
        rows=ctx.data.train_rows[ctx.data.users[ctx.data.train_rows]==u]
        ratings[u,h]=ctx.data.ratings[rows]
    z=x.astype(np.float64)/np.maximum(x.sum(0),1)**.25
    values,vectors=np.linalg.eigh(z@z.T)
    for dim in [32,64,128]:
        # All statistics below are invariant to signs/rotations of the SVD basis.
        emb=(z.T@vectors[:,-dim:])/np.maximum(values[-dim:],1e-12)**.25
        emb/=np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
        for label,q in [('all',x),('last10',weighted_history(ctx,tail=10)),
                        ('last30',weighted_history(ctx,tail=30)),
                        ('liked',(ratings>=4).astype(np.float32)),
                        ('disliked',((ratings>0)&(ratings<=2.5)).astype(np.float32))]:
            profile=q@emb;profile/=np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-12)
            features.append(np.einsum('nd,nd->n',profile[users],emb[items]));names.append(f'svd{dim}_{label}')
        maxsim=np.zeros(len(users),np.float32);lastsim=maxsim.copy();topsim=maxsim.copy()
        for u,h in enumerate(ctx.histories):
            idx=np.flatnonzero(users==u)
            sim=emb[items[idx]]@emb[h[-20:]].T
            maxsim[idx]=sim.max(1);lastsim[idx]=sim[:,-1]
            topsim[idx]=np.sort(sim,axis=1)[:,-min(3,len(h)):].mean(1)
        features.extend([maxsim,lastsim,topsim]);names.extend([f'svd{dim}_recent_max',f'svd{dim}_last',f'svd{dim}_recent_top3'])
    q=weighted_history(ctx,half_life=20)
    denominator=user_knn(x,q,x,20,20)
    rating_sum=user_knn(x,q,ratings,20,20)
    positive=user_knn(x,q,(ratings>=4).astype(np.float32),20,20)
    negative=user_knn(x,q,((ratings>0)&(ratings<=2.5)).astype(np.float32),20,20)
    for name,a in [('neighbor_mean_rating',rating_sum),('neighbor_liked_fraction',positive),('neighbor_disliked_fraction',negative)]:
        features.append((a/np.maximum(denominator,1e-8))[users,items]);names.append(name)
    return np.stack(features,axis=1).astype(np.float32),names
