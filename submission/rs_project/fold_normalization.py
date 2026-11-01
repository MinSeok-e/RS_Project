import numpy as np

from scipy.stats import rankdata

def normalize_fold(features,names,mode):
    keep={'log_user_history','user_mean_rating','item_rating_mean','neighbor_mean_rating','neighbor_liked_fraction','neighbor_disliked_fraction'}
    result=np.empty_like(features)
    for column,name in enumerate(names):
        values=features[:,column].astype(np.float64)
        if name in keep:result[:,column]=values
        elif mode=='zscore':result[:,column]=np.clip((values-values.mean())/max(values.std(),1e-5),-5,5)
        elif mode=='rank':result[:,column]=(rankdata(values,method='average')-.5)/len(values)
        else:raise ValueError('Unknown fold normalizer')
    return result
