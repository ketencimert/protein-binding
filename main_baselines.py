# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:02:04 2022

@author: Mert
"""

import argparse
import random

import numpy as np

import timeit
import torch
from torch import optim

from networks import (CNN_BASELINE, 
                               TN_BASELINE, 
                               FFNN_BASELINE, 
                               TNCNN_BASELINE
                               )

from utils import run_one_epoch, dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #path args
    parser.add_argument('--datadir', default='./data', type=str)
    parser.add_argument('--dataname', default='chip', type=str)

    #device args
    parser.add_argument('--device', default='cpu', type=str)

    #optimization / batch args
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    #model args
    parser.add_argument('--model', default='tncnn', type=str)

    args = parser.parse_args()
    print(args)
    
    seed = 12345
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    
    if args.model not in ['cnn', 'tncnn']:
        #if not cnn based model use the standard seq len at assignment2 = 110
        seq_len = 110
    else:
        seq_len = None
    #initiate the baseline models
    model = {'ffnn':FFNN_BASELINE(),
             'cnn':CNN_BASELINE(),
             'tn':TN_BASELINE(),
             'tncnn':TNCNN_BASELINE(),
             }[args.model].to(args.device)
    #initiate the dataloader
    train_dataloader, validation_dataloader, _ = dataloader(
        datadir=args.datadir,
        dataname=args.dataname,
        seq_len=seq_len if seq_len is not None else model.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        )
    #initiate the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #store the metrics
    stop = 0
    train_accs = []
    val_accs = []
    patience = 10 # for early stopping
    patience_counter = patience
    best_val_loss = np.inf
    
    check_point_filename = 'saves/{}_{}_checkpoint.pt'.format(
        args.model, 
        args.dataname
        ) # to save the best model fit to date
    #train the model
    for epoch in range(args.epochs):
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch(True, train_dataloader, model, optimizer, args.device)
        val_loss, val_acc = run_one_epoch(False, validation_dataloader, model, optimizer, args.device)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if val_loss < best_val_loss: 
            torch.save(model.state_dict(), check_point_filename)
            best_val_loss = val_loss
            patience_counter = patience
        else: 
            patience_counter -= 1
            if patience_counter <= 0: 
                model.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
                break
        elapsed = float(timeit.default_timer() - start_time)
        print("Epoch %i took %.2fs. Train loss: %.4f pr-auc: %.4f. Val loss: %.4f pr-auc: %.4f. Patience left: %i" % 
              (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter ))                
