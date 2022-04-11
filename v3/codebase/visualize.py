# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 23:31:21 2022

@author: Mert
"""
import logomaker
import torch

from codebase.batcher import BedPeaksDataset

from codebase.model import Model

import pandas as pd
import pickle

read = False

if read:

    DATADIR = './data/'
    
    binding_data = pd.read_csv(
        DATADIR + "ENCFF300IYQ.bed.gz",
        sep='\t',
        usecols=range(6),
        names=("chrom","start","end","name","score","strand")
        )
    
    binding_data = binding_data[ ~binding_data['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
    
    binding_data = binding_data.sort_values(
        ['chrom', 'start']
        ).drop_duplicates() # sort so we can interleave negatives
    
    binding_data[:10]
    
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
    
    genome = pickle.load(open(DATADIR+"hg19.pickle","rb"))
    
    best_model = torch.load('saves/best_model_binding.pth')
    
    train_dataset = BedPeaksDataset(
        train_data,
        genome,
        best_model.seq_len
        )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1000,
        num_workers = 0
        )
    
    validation_dataset = BedPeaksDataset(
        validation_data,
        genome,
        best_model.seq_len
        )
    
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=1000
        )
    
    for (x_tr, y_tr) in train_dataloader:
        (x_tr, y_tr)

else:

    best_model = torch.load('saves/best_model_binding.pth')

index = 1

best_model = best_model.cpu()

self = best_model

pwm = self.qw_network.loc.cpu()[index].view(4,20).cpu().detach().numpy()
# pwm = pwm / pwm.sum()
pwm_df = pd.DataFrame(data = pwm.transpose(), columns=("A","C","G","T"))
crp_logo = logomaker.Logo(pwm_df) # CCACCAGG(G/T)GGCG

#GCCCCCTGGTGGC