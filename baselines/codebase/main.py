
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
import timeit
from codebase.batcher import BedPeaksDataset
import torch.nn.functional as F
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

def run_one_epoch(train_flag, dataloader, model, optimizer, device="cuda"):

    torch.set_grad_enabled(train_flag)
    model.train() if train_flag else model.eval() 

    losses = []
    accuracies = []

    for (x,y) in dataloader: # collection of tuples with iterator

        (x, y) = ( x.to(device), y.to(device) ) # transfer data to GPU

        output = model(x) # forward pass
        output = output.squeeze() # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = pr_auc(output, y)
        accuracies.append(accuracy)  
    
    return( np.mean(losses), np.mean(accuracies) )

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

    stop = 0
    device = args.device
    train_accs = []
    val_accs = []
    patience = 10 # for early stopping
    patience_counter = patience
    best_val_loss = np.inf
    
    check_point_filename = 'saves/{}_checkpoint.pt'.format(args.model) # to save the best model fit to date
    for epoch in range(100):
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, device)
        val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_loss < best_val_loss: 
            torch.save(model, check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" % 
              (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter ))                
