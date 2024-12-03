import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import DPBL, ContrastiveLearning
from .optim_schedule import ScheduledOptim

import tqdm
import pdb
import os


class Trainer:
    '''A trainer to train the DPBL model'''

    def __init__(self, dpbl: DPBL, train_dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, batch_size: int = 30, 
                 train_mode: int = 0, load_file: str = None, output_path: str = None, model_name: str=None,
                 config: dict = None, log_path: str = None):

        # Setup cuda device for DPBL training, argument --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print("Device:", self.device)
        self.model_name = model_name

        # This DPBL model will be saved every epoch
        self.dpbl = dpbl.to(self.device)

        self.model = ContrastiveLearning(self.dpbl).to(self.device)
        
        if load_file!=None:
            self.model.dpbl.load_state_dict(torch.load(os.path.join(output_path, load_file), map_location=self.device))

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for DPBL" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train data loader
        self.train_data = train_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.dpbl.hidden, n_warmup_steps=warmup_steps)

        self.log_freq = log_freq
        self.train_mode = train_mode
        self.config = config
        self.batch_size = batch_size
        self.log_path = log_path
        self.load_file = load_file
        self.output_path = output_path

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def iteration(self, epoch, data_loader):
        """
        loop over the data_loader for training and auto save the model every peoch

        Args:
            epoch (int): current epoch index
            data_loader (torch.utils.data.DataLoader): the dataloader for iteration
        """
 
        str_code = "train"
        self.model.train()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0
        avg_loss_1 = 0
        avg_loss_2 = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data_anchor = {key: value.to(self.device) for key, value in data[0].items()}
            data_positive = {key: value.to(self.device) for key, value in data[1].items()}
            data_negative = {key: value.to(self.device) for key, value in data[2].items()}
            
            loss_1, loss_2 = self.model.forward([data_anchor, data_positive, data_negative])
            loss_1 = loss_1.mean()
            loss_2 = loss_2.mean()

            loss = loss_1 + 0.25*loss_2
            self.optim_schedule.zero_grad()
            loss.backward()
            self.optim_schedule.step_and_update_lr()
            avg_loss += loss.item()
            avg_loss_1 += loss_1.item()
            avg_loss_2 += loss_2.item()
            
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": "%.4f" % (avg_loss / (i + 1)),
                "preference_loss": "%.4f" % (avg_loss_1 / (i+1)),
                "noise_loss": "%.4f" % (avg_loss_2 / (i+1))
            }
            
            if i%self.log_freq==0:
                print(post_fix)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

        if self.log_path is not None:
            f = open(os.path.join(self.log_path), 'a')
            f.write("Load file: %s, EP%d, avg_loss=%.4f, avg_loss_1=%.4f, avg_loss_2=%.4f\n" % (self.load_file, epoch, avg_loss / len(data_iter), avg_contrastive/len(data_iter), avg_noise/len(data_iter) ))
            f.close()

    def save(self, epoch):
        """
        Saving the current DPBL model on file_path
        Args:
            epoch (int): current epoch number
        """
        if torch.cuda.device_count() > 1:
            output_name = self.model_name + "_ep" + str(epoch)
            output_path = os.path.join(self.output_path, output_name)
            torch.save(self.model.module.dpbl.state_dict(), output_path)
            print("EP:%d Model Saved on:" % epoch, output_path)

        else:
            output_name = self.model_name + "_ep" + str(epoch)
            output_path = os.path.join(self.output_path, output_name)
            torch.save(self.model.dpbl.state_dict(), output_path)
            print("EP:%d Model Saved on:" % epoch, output_path)
