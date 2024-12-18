import torch.nn as nn


class TimeEmbedding(nn.Embedding):
    def __init__(self, time_segment_num, embed_size=512, padding_idx=-1):
        super().__init__(time_segment_num, embed_size, padding_idx)
