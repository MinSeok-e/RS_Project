import numpy as np

def proportional_targets(targets,positions):
    counts=np.maximum(1,np.ceil(positions/4)).astype(np.int64)
    keep=np.arange(targets.shape[-1])[None,None,:]<counts[...,None]
    return np.where(keep,targets,0)
