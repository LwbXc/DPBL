import torch.nn as nn
import torch
import pdb

from .attention import ExtractorAndDenoising
from .utils import SublayerConnection, PositionwiseFeedForward


class UberlLayer(nn.Module):

    def __init__(self, hidden, attn_heads, feed_forward_hidden, attn_routers, dropout):
        """
        Args:
            hidden (int): hidden size of transformer
            attn_heads (int): head sizes of multi-head attention
            feed_forward_hidden (int): the hidden size for feed_forward layer, usually 4*hidden_size
            dropout (float): dropout rate
        """

        super().__init__()
        self.attention = ExtractorAndDenoising(hidden, attn_heads, attn_routers)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.routers = nn.Parameter(torch.randn(attn_routers, hidden))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x, attn_1, attn_2, hidden_states = self.input_sublayer(x, lambda _x: self.attention.forward(_x, mask), True)
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x), attn_1, attn_2, hidden_states
