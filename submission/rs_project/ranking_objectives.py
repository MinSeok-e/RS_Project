import torch

from .neural_ranker import ranking_loss

def refined_ranking_loss(scores,labels,mask,idcg,objective):
    positive=labels.sum(1)>0
    if not positive.any():return scores.sum()*0
    if objective=='list_idcg':
        return (-(torch.log_softmax(scores,dim=1)*labels).sum(1)/idcg)[positive].mean()
    if objective.startswith('ce_lambda_a'):
        weight=float(objective.removeprefix('ce_lambda_a'))
        return ranking_loss(scores,labels,mask,idcg,'list_ce')+weight*ranking_loss(scores,labels,mask,idcg,'lambda20')
    if objective.startswith('list_top'):
        k=int(objective.removeprefix('list_top'))
        with torch.no_grad():
            negative=(labels==0)&mask
            negatives=scores.masked_fill(~negative,-torch.inf)
            cutoff=torch.topk(negatives,min(k,scores.shape[1]),dim=1).values[:,-1:]
            retained=(labels>0)|(negative&(scores>=cutoff))
        chosen=scores.masked_fill(~retained,-1e4)
        return (-(torch.log_softmax(chosen,dim=1)*labels).sum(1)/labels.sum(1).clamp_min(1))[positive].mean()
    raise ValueError('Unknown refined ranking objective')
