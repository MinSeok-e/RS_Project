from copy import deepcopy

import torch

from torch import nn

from .mixture_ranker import MixtureQueryRanker

class IndependentFeatureMixture(MixtureQueryRanker):
    def __init__(self, n_features, hidden=128, kind='mixture4'):
        super().__init__(n_features, hidden, kind)
        if not self.use_gate:
            raise ValueError('Independent feature experiments require learned gates')
        self.expert_features = nn.ModuleList([deepcopy(self.phi) for _ in self.heads])

    @staticmethod
    def context(h, mask):
        mean = (h*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
        maximum = h.masked_fill(~mask[..., None], -torch.inf).max(1).values
        return torch.cat((mean, maximum), dim=-1)

    def components(self, x, mask):
        gate_context = self.context(self.phi(x), mask)
        gates = (1-self.floor)*torch.softmax(self.gate(gate_context), dim=-1) + self.floor/self.n_experts
        outputs = []
        for feature, head in zip(self.expert_features, self.heads):
            h = feature(x)
            ctx = self.context(h, mask)
            joined = torch.cat((h, ctx[:, None, :].expand(-1, h.shape[1], -1)), dim=-1)
            outputs.append(head(joined))
        return torch.cat(outputs, dim=-1), gates
