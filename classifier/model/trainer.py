import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .model import Predictor
from .optim_schedule import ScheduledOptim

import tqdm
import pdb
import os


class PredictorTrainer:
    '''A trainer to train the predictor for downstream task'''

    def __init__(self, model: Predictor, train_dataloader: DataLoader, 
                 test_dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10, batch_size: int = 30, 
                 log_path: str = None):

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.model.hidden, n_warmup_steps=warmup_steps)

        self.log_freq = log_freq
        self.batch_size = batch_size
        self.log_path = log_path
        self.best_valid = 0
        self.best_test = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def test(self, epoch):
        self.iteration(epoch, self.test_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param train: boolean value of is train or test
        :return: None
        """
 
        if train:
            str_code = "train"
            self.model.train()
        else:
            str_code = "test"
            valid_threshold = int(0.1*len(dataloader)) + 1
            self.model.eval()

        print("epoch:", epoch)
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(dataloader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(dataloader),
                              bar_format="{l_bar}{r_bar}")
        
        avg_loss = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}
            
            nll_loss = self.model.forward(data['embeddings'], data['labels'], train)

            if train:
                nll_loss = nll_loss.mean()
                self.optim_schedule.zero_grad()
                nll_loss.backward()
                self.optim_schedule.step_and_update_lr()
                avg_loss += nll_loss.item()
                
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": "%.4f" % (avg_loss / (i + 1))
                }
            
            else:
                avg_loss += nll_loss.item()
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_acc": "%.4f" % (avg_loss / (self.batch_size*(i + 1)))
                }
                if i==valid_threshold:
                    if avg_loss / (self.batch_size*(i + 1)) > self.best_valid:
                        self.best_valid = avg_loss / (self.batch_size*(i + 1))
                        this_epoch_is_best = True
                    else:
                        this_epoch_is_best = False
            
            if i%self.log_freq==0:
                print(post_fix)

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        if not train:
            if this_epoch_is_best:
                self.best_test = avg_loss / (len(data_iter)*self.batch_size)