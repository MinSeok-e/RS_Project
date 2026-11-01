import torch

def approximate_ndcg_loss(scores,labels,mask,idcg,alpha=1.,top20=True):
    # Smooth rank_i = 1 + sum_{j != i} sigmoid(alpha * (score_j - score_i)).
    probability=torch.sigmoid(alpha*(scores[:,None,:]-scores[:,:,None]))
    ranks=.5+(probability*mask[:,None,:]).sum(2)
    discount=1/torch.log2(ranks+1)
    if top20:discount=discount*torch.sigmoid((20.5-ranks)/2.)
    ndcg=(discount*labels*mask).sum(1)/idcg
    positive=labels.sum(1)>0
    return (1-ndcg[positive]).mean() if positive.any() else scores.sum()*0
