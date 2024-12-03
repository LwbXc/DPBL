import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F

from .dpbl import DPBL
from .utils import GELU

class ContrastiveLearning(nn.Module):
    '''Generate embeddings for the triplet and calculate 
    the contrastive loss according to two contrastive learning strategies'''
    
    def __init__(self, dpbl: DPBL):
        super().__init__()
        self.dpbl = dpbl
        self.contrastive_loss = ContrastiveLoss(self.dpbl.hidden)

    def forward(self, data):

        anchor, hidden_anchor = self.dpbl(data[0])
        positive, hidden_positive = self.dpbl(data[1])
        negative, hidden_negative = self.dpbl(data[2])

        noise_anchor = hidden_anchor[:, :, 0:1, :]
        noise_positive = torch.cat([hidden_negative[:, :, 0:1, :], hidden_positive[:, :, 0:1, :]], dim=2)
        noise_negative = hidden_anchor[:, :, 1:, :]

        return self.contrastive_loss(anchor.unsqueeze(1), positive.unsqueeze(1), negative.unsqueeze(1)), self.contrastive_loss(noise_anchor, noise_positive, noise_negative)

class ContrastiveLoss(nn.Module):

    def __init__(self, hidden):
        super(ContrastiveLoss, self).__init__()
        self.output_linear1 = nn.Linear(hidden, hidden)
        self.activation = GELU()
        self.output_linear2 = nn.Linear(hidden, hidden)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-9)

    def forward(self, anchor, positive, negative):
        return self.contrastive_loss(anchor, positive, negative)
    
    def contrastive_loss(self, origin, positive, negative, tau=0.1):
        sim_p = self.cos(origin, positive)
        sim_n = self.cos(origin, negative)

        sim_p = torch.exp(sim_p/tau).sum(dim=-1)
        sim_n = torch.exp(sim_n/tau).sum(dim=-1)

        sc = -torch.log(sim_p /(sim_p + sim_n))
        return sc