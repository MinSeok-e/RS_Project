import numpy as np

def history_model_quality(histories,old_lengths,rankings):
    discounts=1/np.log2(np.arange(2,22));ndcg=np.zeros(len(histories));available=np.zeros(len(histories),bool)
    for u,h in enumerate(histories):
        start=int(old_lengths[u]);end=min(len(h),start+max(1,int(np.ceil(start*.25))))
        if end<=start:continue
        truth=set(h[start:end]);available[u]=True
        hits=np.array([item in truth for item in rankings[u]],dtype=float)
        ndcg[u]=np.dot(hits,discounts)/discounts[:min(len(truth),20)].sum()
    return ndcg,available
