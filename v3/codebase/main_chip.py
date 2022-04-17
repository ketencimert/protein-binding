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
from codebase.utils import save

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
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    #model, encoder-decoder args
    parser.add_argument('--num_chunks', default=5, type=int)
    parser.add_argument('--max_pool_factor', default=4, type=int)
    parser.add_argument('--nchannels', default=[4, 32, 32], type=list)
    parser.add_argument('--n_hidden', default=32, type=int)
    parser.add_argument('--dropout', default=0.2, type=int)
    parser.add_argument('--num_motifs', default=2, type=int)
    parser.add_argument('--kernel_size', default=20, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--use_z', default=False, type=bool)
    parser.add_argument('--metric', default='pr', type=str)
   
    #data, fold, tune, metric args
    parser.add_argument('--early_stop', default=10, type=int)
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

    train_dataset = BedPeaksDataset(
        train_data,
        genome,
        model.seq_len
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
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
        batch_size=args.batch_size
        )

    decoder = filter(
        lambda kv: kv[0] in ['py_xwz_network'],
        model.named_parameters()
        )

    decoder = [var[1] for var in decoder]

    others = filter(
        lambda kv: kv[0] not in ['py_xwz_network'],
        model.named_parameters()
        )

    others = [var[1] for var in others]

    optimizer = optim.Adam([
            {'params': decoder,
             'lr': args.lr},
            {'params': others,
             'lr': args.lr},
        ]
        )

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

                save(best_model, 'binding_chip')
        
        print('Epoch : {} Best Score : {}'.format(epoch, best_scores[-1]))