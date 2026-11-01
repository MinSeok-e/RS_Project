import torch

from .common import weighted_history

from .filter_sequence import inference_tokens

def views(model,ctx):
    model.eval();tokens=torch.as_tensor(inference_tokens(ctx.histories,model.positions.num_embeddings),device='cuda')
    with torch.no_grad():
        result={'prediction':torch.cat([model.score(tokens[j:j+128]) for j in range(0,len(tokens),128)]).cpu().numpy()}
        emb=torch.nn.functional.normalize(model.items.weight[1:],dim=1)
        for label,q in [('all',ctx.x),('recent20',weighted_history(ctx,half_life=20))]:
            profile=torch.as_tensor(q,device='cuda')@emb;profile=torch.nn.functional.normalize(profile,dim=1)
            result['profile_'+label]=(profile@emb.T).cpu().numpy()
    return result
