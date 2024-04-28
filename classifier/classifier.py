from model import ClassifierDataset, PredictorTrainer, Predictor
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import argparse
import json
import pdb
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_embeddings", type=str, help="path of the embeddings of train dataset")
    parser.add_argument("--train_labels", type=str, help="path of the labels of train dataset")
    parser.add_argument("--test_embeddings", type=str, help="path of the embeddings of test dataset")
    parser.add_argument("--test_labels", type=str, help="path of the labels of test dataset")
    
    parser.add_argument("--embedding_save", type=str, help="path to save embeddings")
    parser.add_argument("--result_save", type=str, help="path of logs")

    parser.add_argument("-hs", "--hidden", type=int, default=128, help="hidden size")
    parser.add_argument("--layers", type=int, default=2, help="number of layers in the predictor")

    parser.add_argument("-b", "--batch_size", type=int, default=128, help="number of batch_size")
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

    train_dataset = ClassifierDataset(args.train_embeddings, args.train_labels)
    weight = np.array(json.load(open(args.train_labels, 'r')), dtype=float)
    positive = (weight>0).sum()
    negative = (weight==0).sum()
    weight[weight>0] = negative
    weight[weight==0] = positive
    sampler = WeightedRandomSampler(weight, len(weight))


    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)

    test_dataset = ClassifierDataset(args.test_embeddings, args.test_labels)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) 
    
    
    print("Building model")
    predictor = Predictor(hidden=args.hidden, n_layers=args.layers, n_class=2)

    print("Creating Trainer")
    trainer = PredictorTrainer(predictor, train_dataloader=train_data_loader, 
                        test_dataloader=test_data_loader,
                        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                        with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, batch_size = args.batch_size,
                        log_path = args.result_save)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        
        with torch.no_grad():
            trainer.test(epoch)

    print("Best valid", trainer.best_valid, "Best test", trainer.best_test)
    with open(args.result_save, 'a') as f:
        f.write("Model {}, best valid {}, best test {}\n".format(
                        args.model_name, str(trainer.best_valid), str(trainer.best_test)))


if __name__ == '__main__':
    main()
