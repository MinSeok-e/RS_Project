import numpy as np

from scipy import sparse

from .models import retain_row_topk

def cosine_graph(x,k=100):
    x=sparse.csr_matrix(x,dtype=np.float32);norm=np.sqrt(np.asarray(x.power(2).sum(0)).ravel()).clip(1)
    blocks=[]
    for start in range(0,x.shape[1],256):
        block=(x[:,start:start+256].T@x).tocsr()
        block=sparse.diags(1/norm[start:start+block.shape[0]])@block@sparse.diags(1/norm)
        for j in range(block.shape[0]):
            a,b=block.indptr[j:j+2];block.data[a:b][block.indices[a:b]==start+j]=0
        block.eliminate_zeros();blocks.append(retain_row_topk(block,k))
    return sparse.vstack(blocks,format='csr')

def propagate(probability,graph,eligible):
    result=(graph.T@probability.T).T
    result=np.where(eligible,np.maximum(result,0),0).astype(np.float32)
    mass=result.sum(1,keepdims=True)
    return np.divide(result,mass,out=probability.copy(),where=mass>0)
