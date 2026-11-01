import numpy as np

def transform(scores, eligible, method):
    a=np.asarray(scores,dtype=np.float64)
    # Rerankers explicitly mark candidates outside their retrieval pool with -1e6.
    # That sentinel must not determine the normalization of actual model scores.
    known=eligible & (a!=-1e6)
    if (known.sum(1)==0).any():raise ValueError('No scored candidates')
    if method=='zscore':
        mean=(a*known).sum(1,keepdims=True)/known.sum(1,keepdims=True)
        var=((a-mean)**2*known).sum(1,keepdims=True)/known.sum(1,keepdims=True)
        out=(a-mean)/np.sqrt(np.maximum(var,1e-20))
        floor=np.where(known,out,np.inf).min(1,keepdims=True)-1
        out=np.where(known,out,floor)
        return out.astype(np.float32)
    masked=a.copy();masked[~eligible]=-np.inf
    order=np.argsort(-masked,axis=1,kind='stable')
    ranks=np.empty(order.shape,dtype=np.int32)
    np.put_along_axis(ranks,order,np.broadcast_to(np.arange(1,a.shape[1]+1),a.shape),axis=1)
    out=1/(60+ranks.astype(np.float32));out[~known]=0
    return out
