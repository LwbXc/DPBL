import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import Uberl
from .optim_schedule import ScheduledOptim

import pdb
import json
import os
import time
import tqdm


class PrintAttention:

    def __init__(self, uberl: Uberl, dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, train_mode: int = 0, 
                 load_file: str = None, output_path: str = None, config: dict = None,
                 embedding_path: str = None, remarks: str = ""):

        # Setup cuda device for Uberl inference, argument --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.model = uberl.to(self.device)
        if load_file!=None:
            print("Load model from", os.path.join(output_path, load_file))
            self.model.load_state_dict(torch.load(os.path.join(output_path, load_file), map_location=self.device))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for Uberl" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # Setting the test data loader
        self.data_loader = dataloader
        self.load_file = load_file
        self.train_mode = train_mode
        self.extractor_attn_path = embedding_path[:-5] + "_extractor.json"
        self.denoising_attn_path = embedding_path[:-5] + "_denoising.json"
        self.event_path = embedding_path[:-5] + "_event.json"
        self.remarks = remarks

    def iteration(self):
        """
        loop over the data_loader and print attention weights
        """
        self.model.eval()
        # pdb.set_trace()
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.data_loader),
                              desc="Print Attention Weights",
                              total=len(self.data_loader),
                              bar_format="{l_bar}{r_bar}")

        # pdb.set_trace()

        all_attn_1 = []
        all_attn_2 = []
        all_events = []
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            extractor_attn, denoising_attn = self.model.forward(data, print_attn=True)           
            
            extractor_attn = extractor_attn.cpu().detach().numpy().tolist() 
            denoising_attn = denoising_attn.cpu().detach().numpy().tolist() 

            events = data['event'].cpu().detach().numpy().tolist()
            
            all_attn_1.append(extractor_attn)
            all_attn_2.append(denoising_attn)
            all_events.append(events)

        with open(self.extractor_attn_path,'w') as f:
            json.dump(all_attn_1, f)
        
        with open(self.denoising_attn_path,'w') as f:
            json.dump(all_attn_2, f)

        with open(self.event_path, 'w') as f:
            json.dump(all_events, f)
