import os
import torch
import json
import csv
import pdb
from torch.utils.data import Dataset
import numpy as np
import random
import math
import copy

class DatasetTest(Dataset):

    def __init__(self, config, file_path):
        """
        The dataloader to load dataset for inference

        Args:
            config (dict): hyperparameters
            file_path (str): path of the dataset to be loaded
        """
        self.config = config
        self.lines = csv.reader(open(file_path, 'r'))
        self.lines = list(self.lines)
        self.file_path = file_path
        self.max_length = config['max_len']
    
    def __len__(self):
        return len(self.lines)

    def preprocess(self, line):
        event_and_time = line[2:]
        if len(event_and_time)>2*self.max_length:
            event_and_time = event_and_time[:2*self.max_length]
        event = []
        time = []
        for _i, x in enumerate(event_and_time):
            if _i%2==0:
                time.append(int(x))
            else:
                event.append(int(x))
        time_difference = [0] + [time[i+1]-time[i] for i in range(len(time)-1)]
        
        time_difference += [self.config['time_embed_num']+1]*(self.max_length-len(time_difference))
        event += [self.config['max_event_num']+1]*(self.max_length-len(event))
        mask = [1]*len(time) + [0]*(self.max_length-len(time))
        
        time_difference = torch.LongTensor(time_difference)
        event = torch.LongTensor(event)
        mask = torch.LongTensor(mask)
        return time_difference, event, mask

    def __getitem__(self, item):

        line_anchor = self.lines[item]
        
        time_anchor, event_anchor, mask_anchor = self.preprocess(line_anchor)

        output_anchor = {
            "time": time_anchor,
            "event": event_anchor,
            "mask": mask_anchor
        }

        return output_anchor
