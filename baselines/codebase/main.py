
import argparse
from copy import deepcopy
import gc

import random
import pickle
import pandas as pd

import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
from torch import optim
from tqdm import tqdm

from codebase.batcher import BedPeaksDataset

from codebase.networks import *

from codebase.utils import save, save_runs


from codebase.utils import (
    kl_divergence_1,
    kl_divergence_2,
    rmse,
    accuracy,
    roc_auc,
    pr_auc
    )

DATADIR = './data/'

def evaluate(model, batcher):

    with torch.no_grad():

        x_va, y_va = batcher.next('valid')

        elbo, score = model(x_va, y_va)

    return elbo, score

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #device args
    parser.add_argument('--device', default='cpu', type=str)

    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--model', default='cnn', type=str)

    parser.add_argument('--epochs', default=200, type=int)

    args = parser.parse_args()

    print(args)

    seed = 12345

    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    genome = pickle.load(open(DATADIR+"hg19.pickle","rb"))

    binding_data = pd.read_csv(
        DATADIR + "ENCFF300IYQ.bed.gz",
        sep='\t',
        usecols=range(6),
        names=("chrom","start","end","name","score","strand")
        )

    binding_data = binding_data[~binding_data['chrom'].isin(["chrX","chrY"])]
    binding_data = binding_data.sort_values(
        ['chrom', 'start']).drop_duplicates()

    test_chromosomes = ["chr1"]
    test_data = binding_data[ binding_data['chrom'].isin( test_chromosomes ) ]

    validation_chromosomes = ["chr2","chr3"]
    validation_data = binding_data[
        binding_data['chrom'].isin(validation_chromosomes)
        ]

    train_chromosomes = ["chr%i" % i for i in range(4, 22+1)]
    train_data = binding_data[
        binding_data['chrom'].isin(train_chromosomes)
        ]

    model = {'ffnn':FeedforwardNN(),
             'cnn':CNN_1d(),
             'ann':AttentionNN(),
             }[args.model].to(args.device)
    
    model.seq_len = 110
    
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

    loglikelihoods = []

    stop = 0

    for epoch in range(args.epochs):
        
        model.train()
        
        for (x_tr, y_tr) in train_dataloader:
            
            x_tr, y_tr = x_tr.to(args.device), y_tr.to(args.device)
            
            optimizer.zero_grad()

            y_pred = torch.sigmoid(model(x_tr))
            
            loglikelihood = -Bernoulli(probs=y_pred).log_prob(y_tr).mean()
            
            loglikelihood.backward()

            optimizer.step()

            loglikelihoods.append(loglikelihood)
        
        with torch.no_grad():
        
            model.eval()
            
            scores_ = []
            
            for (x_va, y_va) in validation_dataloader:
            
                x_va, y_va = x_va.to(args.device), y_va.to(args.device)
                
                y_pred = torch.sigmoid(model(x_va))
                
                score = pr_auc(y_pred, y_va)
                
                scores_.append(score)
                
            scores.append(np.mean(scores_))

            best_scores.append(max(scores))

            mean_scores.append(np.mean(scores))

            if scores[-1] >= max(scores):

                best_model = deepcopy(model)

                save(best_model, 'binding_{}'.format(args.model))
        
        print(epoch, best_scores[-1])