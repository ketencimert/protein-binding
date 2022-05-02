# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:02:04 2022

@author: Mert
"""

import argparse
from copy import deepcopy

import random
import numpy as np
import torch

from torch import optim
from tqdm import tqdm

from utils import save, dataloader
from model import Model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #path args
    parser.add_argument('--datadir', default='./data', type=str)
    parser.add_argument('--dataname', default='atac', type=str)

    #device args
    parser.add_argument('--device', default='cuda', type=str)

    #optimization / batch args
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    #model args
    parser.add_argument('--num_chunks', default=5, type=int)
    parser.add_argument('--max_pool_factor', default=4, type=int)
    parser.add_argument('--nchannels', default=[4, 32, 32], type=list)
    parser.add_argument('--n_hidden', default=32, type=int)
    parser.add_argument('--dropout', default=0.2, type=int)
    parser.add_argument('--num_motifs', default=10, type=int)
    parser.add_argument('--kernel_size', default=20, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--use_z', default=True, type=bool)
    parser.add_argument('--metric', default='pr', type=str)

    args = parser.parse_args()
    print(args)
    
    seed = 12345
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    model = Model(
            num_chunks=args.num_chunks,
            max_pool_factor=args.max_pool_factor,
            nchannels=args.nchannels,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            num_motifs=args.num_motifs,
            kernel_size=args.kernel_size,
            stride=args.stride,
            use_z=args.use_z,
            metric=args.metric,
        ).to(args.device)

    train_dataloader, validation_dataloader, _ = dataloader(
        datadir=args.datadir,
        dataname=args.dataname,
        seq_len=model.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    mean_scores = []
    best_scores = []
    scores = []
    elbos = []
    stop = 0

    for epoch in tqdm(range(args.epochs)):

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
                save(
                    best_model, '{}_checkpoint'.format(
                        args.dataname
                        )
                    )

        print('Epoch : {} Best Score : {}'.format(epoch, best_scores[-1]))