import pdb
import torch
import torch.nn as nn

from .layer import DPBLLayer
from .embedding import DPBLEmbedding
from .utils import LayerNorm, GELU
import torch.nn.functional as F

class DPBL(nn.Module):

    def __init__(self, config, hidden=128, n_layers=2, attn_heads=12, preference_num=12, dropout=0.1):

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # embedding for DPBL, sum of positional, time interval, action embeddings
        self.embedding = DPBLEmbedding(config, embed_size=hidden, head_num=attn_heads)

        self.layers = nn.ModuleList(
            [DPBLLayer(hidden, attn_heads, hidden, preference_num, dropout) for _ in range(n_layers)])


        self.output_linear1 = nn.Linear(hidden, int(hidden/4))
        self.activation = GELU()
        self.output_linear2 = nn.Linear(int(hidden/4), 1)

        self.norm = LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data, print_attn=False):

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(data)

        # print the attention scores in the 1st layer for train_mode==2
        attn_1 = None
        attn_2 = None
        
        hidden_states = []

        # running over multiple transformer blocks
        for layer in self.layers:
            if attn_1 is None:
                x, attn_1, attn_2, hidden = layer.forward(x, data['mask'])
            else:
                x, _, _, hidden = layer.forward(x, data['mask'])
            hidden_states.append(hidden)

        hidden_states = torch.stack(hidden_states)
       
        scores = self.output_linear2(self.dropout(self.activation(self.output_linear1(x))))
        scores = scores.permute(0, 2, 1)
        scores = scores.masked_fill(data['mask'].unsqueeze(1) == 0, -1e9)
        
        weight = F.softmax(scores, dim=-1)
        output = torch.matmul(weight, x).squeeze(1)

        if print_attn:
            return attn_1, attn_2
        else:
            return self.norm(output), hidden_states