import numpy as np

def calibrated(scores,eligible,kind,parameter,power=1.):
    a=np.asarray(scores,dtype=np.float64);known=eligible&(a!=-1e6)
    if not known.any(1).all():raise ValueError('No candidates')
    if kind=='probability':
        mean=(a*known).sum(1,keepdims=True)/known.sum(1,keepdims=True)
        var=((a-mean)**2*known).sum(1,keepdims=True)/known.sum(1,keepdims=True)
        logits=(a-mean)/np.sqrt(np.maximum(var,1e-20))/parameter
        logits[~known]=-np.inf;logits-=logits.max(1,keepdims=True)
        values=np.exp(logits);values/=values.sum(1,keepdims=True)
    elif kind=='rank':
        a=np.where(known,a,-np.inf);order=np.argsort(-a,axis=1,kind='stable')
        ranks=np.empty_like(order)
        np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,a.shape[1]+1),a.shape),axis=1)
        values=1/(parameter+ranks.astype(float))**power;values[~known]=0
    else:raise ValueError('Unknown fusion')
    return values.astype(np.float32)
