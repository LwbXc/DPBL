import pdb
import torch
import torch.nn as nn
from .gelu import GELU

class Predictor(nn.Module):
    '''A simple predictor for downstream tasks'''

    def __init__(self, hidden=128, n_layers=2, n_class=2):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_class = n_class

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden),
                          GELU(),
                          nn.Linear(hidden, hidden)
                          )
                                     for _ in range(n_layers)])

        self.projection_head = nn.Linear(hidden, n_class)
        self.nll = nn.NLLLoss(ignore_index=2)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, data, labels, train):
        # pdb.set_trace()
        for layer in self.layers:
            data = layer(data)
        output = self.projection_head(data)
        if train:
            output = self.logsoftmax(output)
            loss = self.nll(output, labels.squeeze(1))
            # pdb.set_trace()
            return loss
        else:
            output = torch.argmax(output, dim=-1)
            if_true = (output==labels.squeeze(-1))
            output = if_true.sum()
            
            return output