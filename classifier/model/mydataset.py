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

class ClassifierDataset(Dataset):

    def __init__(self, embedding_path, label_path):
        """
        Input:
        embedding_path (str): file path of the embeddings to be loaded
        label_path (str): file path of the labels to be loaded
        """
        self.embeddings = json.load(open(embedding_path, 'r'))
        self.labels = json.load(open(label_path, 'r'))

    def __len__(self):
        assert len(self.embeddings)==len(self.labels)
        return len(self.embeddings)

    def __getitem__(self, item):

        embeddings = self.embeddings[item]
        labels = [self.labels[item]]

        embeddings = torch.FloatTensor(embeddings)
        labels = torch.LongTensor(labels)
        labels[labels>1] = 1

        data = {
            "embeddings": embeddings,
            "labels": labels
        }

        return data