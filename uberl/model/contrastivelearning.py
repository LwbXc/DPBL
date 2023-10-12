import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F

from .uberl import Uberl
from .utils import GELU

class ContrastiveLearning(nn.Module):
    
    def __init__(self, uberl: Uberl):
        """
        :param bert: BERT model which should be trained
        """

        super().__init__()
        self.uberl = uberl
        self.contrastive_loss = ContrastiveLoss(self.uberl.hidden)

    def forward(self, data):

        anchor, hidden_anchor = self.uberl(data[0])
        positive, hidden_positive = self.uberl(data[1])
        negative, hidden_negative = self.uberl(data[2])

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