
import argparse
from copy import deepcopy
import gc

import random
import pickle
import pandas as pd

import numpy as np
import torch

from torch import optim
from tqdm import tqdm

from codebase.batcher import BedPeaksDataset

from codebase.model import Model

from codebase.utils import save, save_runs

from torch import nn

DATADIR = './data/'

def evaluate(model, batcher):

    with torch.no_grad():

        x_va, y_va = batcher.next('valid')

        elbo, score = model(x_va, y_va)

    return elbo, score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #device args

    parser.add_argument('--device', default='cuda', type=str)

    #optimization args

    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--epochs', default=600, type=int)

    parser.add_argument('--batch_size', default=1024, type=int)

    #model, encoder-decoder args

    parser.add_argument('--norm', default='layernorm', type=str)

    parser.add_argument('--activation', default='elu', type=str)

    parser.add_argument('--prior_dist', default='normal', type=str)

    parser.add_argument('--dropout_input', default=0.5, type=float)

    parser.add_argument('--dropout_output', default=0.0, type=float)

    parser.add_argument('--amortize_bias', default=False, type=bool)

    parser.add_argument('--dropout_intermediate', default=0.5, type=float)

    parser.add_argument('--generative_scale_grad', default=False, type=bool)

    parser.add_argument('--layer_size', default=[48, 32], type=int, nargs='+')

    parser.add_argument('--variational_network', default='feedforward_3', type=str)

    #data, fold, tune, metric args

    parser.add_argument('--folds', default=1, type=int)

    parser.add_argument('--auc', default='pr', type=str)

    parser.add_argument('--early_stop', default=10, type=int)

    parser.add_argument('--visualize', default=True)

    parser.add_argument('--check_early_stop', default=1000, type=int)

    args = parser.parse_args()

    print(args)

    seed = 12345

    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    genome = pickle.load(open(DATADIR+"hg19.pickle","rb"))

    atac_data = pd.read_csv(
        DATADIR + "ATAC_data.bed.gz", sep='\t', names=("chrom", "start", "end")
    )

    atac_data = atac_data.sort_values(['chrom', 'start'])  # actually already sorted but why not

    validation_chromosomes = ["chr3", "chr4"]
    validation_data = atac_data[
        atac_data['chrom'].isin(validation_chromosomes)
    ]

    train_data = atac_data[
        ~atac_data['chrom'].isin(validation_chromosomes)
    ]

    model = Model(num_motifs=100, kernel_size=20).to(args.device)

    train_dataset = BedPeaksDataset(
        train_data,
        genome,
        model.seq_len
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1000,
        num_workers = 0
        )

    validation_dataset = BedPeaksDataset(
        validation_data,
        genome,
        model.seq_len
        )

    del genome
    gc.collect()

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1000
        )

    generative_scale = filter(
        lambda kv: kv[0] in ['generative_scale'],
        model.named_parameters()
        )

    generative_scale = [var[1] for var in generative_scale]

    others = filter(
        lambda kv: kv[0] not in ['generative_scale'],
        model.named_parameters()
        )

    others = [var[1] for var in others]

    optimizer = optim.Adam([
            {'params': generative_scale,
             'lr': args.lr/50.},
            {'params': others,
             'lr': args.lr},
        ]
        )

    mean_scores = []

    best_scores = []

    scores = []

    elbos = []

    stop = 0

    for epoch in range(args.epochs):
        
        model.train()
        
        for (x_tr, y_tr) in train_dataloader:
            
            x_tr, y_tr = x_tr.to(args.device), y_tr.to(args.device)
            
            optimizer.zero_grad()

            elbo, score, y_pred = model(x_tr, y_tr)

            elbo.backward()

            optimizer.step()

            elbos.append(elbo)

        with torch.no_grad():
            
            model.eval()
            
            # w, _ = model.qw_network()
            
            # print(nn.Softmax(-1)(model.pz_wyx_network(model, w,y_tr, x_tr).mean(0)))
            
            scores_ = []
            
            for (x_va, y_va) in validation_dataloader:
            
                x_va, y_va = x_va.to(args.device), y_va.to(args.device)
                
                elbo, score, y_pred = model(x_va, y_va)
                
                scores_.append(score)
                
            scores.append(np.mean(scores_))

            best_scores.append(max(scores))

            mean_scores.append(np.mean(scores))

            if scores[-1] >= max(scores):

                best_model = deepcopy(model)

                save(best_model, 'binding')
        
        print(epoch, best_scores[-1])