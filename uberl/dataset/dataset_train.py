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

class DatasetTrain(Dataset):

    def __init__(self, config, file_path):
        """
        The dataloader to load triplet for pre-training

        Args:
            config (dict): hyperparameters
            file_path (str): path of the dataset to be loaded
        """
        self.config = config
        dataset = csv.reader(open(file_path, 'r'))
        self.lines = []
        for i, item in enumerate(dataset):
            if len(item)>62:
                self.lines.append(item)
        self.file_path = file_path
        self.max_length = config['max_len']
    
    def __len__(self):
        return len(self.lines)

    def sample(self, line, p):
        """
        Downsample the behavior sequence

        Args:
            line (list): a behavior sequence
            p (float): the downsampling rate
        """
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
        time = np.array(time)
        event = np.array(event)
        mask = np.array([random.random() for i in range(len(time))])
        time = (time[mask<p]).tolist()
        event = (event[mask<p]).tolist()
        time_difference = [0] + [time[i+1]-time[i] for i in range(len(time)-1)]
        
        tmp = torch.LongTensor(time)
        mask = [1]*len(time) + [0]*(self.max_length-len(time))
        time_difference += [self.config['time_embed_num']+1]*(self.max_length-len(time))
        event += [self.config['max_event_num']+1]*(self.max_length-len(event))
        
        time_difference = torch.LongTensor(time_difference)
        event = torch.LongTensor(event)
        mask = torch.LongTensor(mask)
        return time_difference, event, mask

    def __getitem__(self, item):

        line_anchor = self.lines[item]
        
        q = random.random()
        index = int(q*(len(self.lines)-1))
        line_negative = self.lines[index]
        
        time_anchor, event_anchor, mask_anchor = self.sample(line_anchor, 0.8+random.random()*0.1)
        time_positive, event_positive, mask_positive = self.sample(line_anchor, 0.8+random.random()*0.1)
        time_negative, event_negative, mask_negative = self.sample(line_negative, 0.8+random.random()*0.1)


        output_anchor = {
            "time": time_anchor,
            "event": event_anchor,
            "mask": mask_anchor
        }

        output_positive = {
            "time": time_positive,
            "event": event_positive,
            "mask": mask_positive
        }

        output_negative = {
            "time": time_negative,
            "event": event_negative,
            "mask": mask_negative
        }


        return output_anchor, output_positive, output_negative
