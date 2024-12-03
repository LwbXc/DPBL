import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import DPBL
from .optim_schedule import ScheduledOptim

import pdb
import json
import os
import time
import tqdm


class Inference:

    def __init__(self, dpbl: DPBL, dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, train_mode: int = 0, 
                 load_file: str = None, output_path: str = None, config: dict = None,
                 embedding_path: str = None, remarks: str = ""):

        # Setup cuda device for DPBL inference, argument --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.model = dpbl.to(self.device)
        if load_file!=None:
            print("Load model from", os.path.join(output_path, load_file))
            self.model.load_state_dict(torch.load(os.path.join(output_path, load_file), map_location=self.device))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for DPBL" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        # Setting the test data loader
        self.data_loader = dataloader
        self.load_file = load_file
        self.train_mode = train_mode
        self.embedding_path = embedding_path
        self.remarks = remarks

    def iteration(self):
        """
        loop over the data_loader for inference
        """
        self.model.eval()
        data_iter = tqdm.tqdm(enumerate(self.data_loader),
                              desc="Inference",
                              total=len(self.data_loader),
                              bar_format="{l_bar}{r_bar}")

        hidden_embeddings = []
        noise_list = []
        stime = time.time()

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            hidden_embedding, noise = self.model.forward(data)            
            hidden_embeddings += hidden_embedding.cpu().detach().numpy().tolist()
            noise_list += noise.cpu().detach().numpy().tolist()

        with open (self.embedding_path,'w') as f:
            json.dump([hidden_embeddings, noise_list], f)
