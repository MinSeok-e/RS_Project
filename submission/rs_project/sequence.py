import torch

from torch import nn

class GRURec(nn.Module):
    def __init__(self,n_items,maxlen=100,dim=64,blocks=1,heads=1,dropout=.2):
        super().__init__()
        self.item_embedding=nn.Embedding(n_items+1,dim,padding_idx=0)
        self.drop=nn.Dropout(dropout)
        self.gru=nn.GRU(dim,dim,blocks,batch_first=True,dropout=dropout if blocks>1 else 0)
        self.norm=nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        with torch.no_grad():self.item_embedding.weight[0].zero_()

    def encode(self,tokens):
        return self.norm(self.gru(self.drop(self.item_embedding(tokens)))[0])

    def score(self,tokens):
        h=self.encode(tokens)
        last=(tokens>0).sum(1)-1
        return h[torch.arange(len(tokens),device=tokens.device),last]@self.item_embedding.weight[1:].T
