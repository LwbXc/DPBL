import torch.nn as nn
import torch
import pdb
from .position import PositionalEmbedding
from .timestamp import TimeEmbedding
from .event import EventEmbedding

class UberlEmbedding(nn.Module):

    def __init__(self, config, embed_size, head_num, dropout=0.1):
        super().__init__()

        self.timestamp = TimeEmbedding(time_segment_num=config["time_embed_num"]+4, embed_size=embed_size, padding_idx=config["time_embed_num"]+2)
        self.event = EventEmbedding(event_num=config["max_event_num"]+3, embed_size=embed_size, padding_idx=config["max_event_num"]+2)
    
        self.position = PositionalEmbedding(d_model=int(embed_size), max_len=config['max_len']+2)
        self.dropout = nn.Dropout(p=dropout)
        self.config = config
        self.embed_size = embed_size

    def forward(self, data):

        data['time'][data['time']>self.config["time_embed_num"]] = self.config["time_embed_num"]
                
        x = self.timestamp(data['time']) + self.event(data['event']) + self.position(data['event'])
        
        return self.dropout(x)