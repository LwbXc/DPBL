import torch.nn as nn
from .gelu import GELU
import pdb

class FeedForwardInteraction(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_in, d_out, dropout=0.1):
        super(FeedForwardInteraction, self).__init__()
        self.w_1 = nn.Linear(d_in, 2*d_in)
        self.w_2 = nn.Linear(2*d_in, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        # pdb.set_trace()
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
