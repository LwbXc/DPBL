from uberl.dataset import *
from uberl.model import Uberl
from uberl.trainer import Trainer, Inference, PrintAttention
from torch.utils.data import DataLoader
import torch
import argparse
import json
import pdb
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_mode", type=int, default=0, help="0)pretrain, 1)inference, 2)save the attention weight of Uberl")
    parser.add_argument("--train_dataset", type=str, help="path of train dataset")
    parser.add_argument("--test_dataset", type=str, help="path of test_dataset", default="")
    parser.add_argument("--model_save", type=str, help="path to save model")
    parser.add_argument("--model_name", type=str, help="the file name to save the model with", default='uberl')
    parser.add_argument("--embedding_save", type=str, help="path to save embeddings or attention weights in mode 1 and mode 2")
    parser.add_argument("--log_path", type=str, help="directory path to save logs")
    parser.add_argument("--load_file", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config.json")

    parser.add_argument("--hidden", type=int, default=128, help="hidden size")
    parser.add_argument("--layers", type=int, default=2, help="number of layers of Uberl")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--preference_num", type=int, default=16, help="number of preference distributions")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1], help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    if args.train_mode == 0:    
        train_dataset = DatasetTrain(config, args.train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
        print("Building Uberl model")
        uberl = Uberl(config, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, preference_num=args.preference_num)

        print("Creating Uberl Trainer")
        trainer = Trainer(uberl, train_dataloader=train_data_loader, 
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, batch_size = args.batch_size,
                            train_mode = args.train_mode, load_file = args.load_file, output_path = args.model_save, model_name=args.model_name, config = config, 
                            log_path = args.log_path)
    
        print("Training Start")
        for epoch in range(args.epochs):
            trainer.train(epoch)
            trainer.save(epoch)

    elif args.train_mode == 1:
        print("Creating Dataloader")
        if args.test_dataset:
            test_dataset = DatasetTest(config,  args.test_dataset)
            test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        else:
            print("For inferece mode, test dataset should be provided")
            exit()

        print("Building Uberl model")
        uberl = Uberl(config, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, preference_num=args.preference_num)

        print("Creating Uberl Inference")
        trainer = Inference(uberl, dataloader=test_data_loader, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), 
                            weight_decay=args.adam_weight_decay, with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, 
                            train_mode = 1, load_file = args.load_file, output_path = args.model_save, config = config,
                            embedding_path = args.embedding_save)
        
        with torch.no_grad():
            trainer.iteration()

    elif args.train_mode == 2:
        print("Creating Dataloader")
        if args.test_dataset:
            test_dataset = DatasetTest(config,  args.test_dataset)
            test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        print("Building Uberl model")
        uberl = Uberl(config, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, preference_num=args.preference_num)

        print("Creating Uberl Trainer")
        trainer = PrintAttention(uberl, dataloader=test_data_loader, lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), 
                            weight_decay=args.adam_weight_decay, with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, 
                            train_mode = 1, load_file = args.load_file, output_path = args.model_save, config = config,
                            embedding_path = args.embedding_save)
        
        with torch.no_grad():
            trainer.iteration()


if __name__ == '__main__':
    main()
