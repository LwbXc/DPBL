import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, print_attn=False):
        "Apply residual connection to any sublayer with the same size."
        if print_attn:
            output_x, attn_s, attn_r, buffer = sublayer(self.norm(x))
            return x + self.dropout(output_x), attn_s, attn_r, buffer
        else:
            return x + self.dropout(sublayer(self.norm(x)))
