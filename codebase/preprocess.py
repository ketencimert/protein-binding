
import argparse
import pickle

import numpy as np
import pandas as pd

from generator import Generator
from sklearn.preprocessing import MultiLabelBinarizer
import category_encoders as ce

INPUT_FN = './data/clip/'

def load_rna_binding(dataname):
    
    y_scale = 1
    
    task = 'classification'
    
    path = '{}/30000/training_sample_0/sequences.fa'.format(dataname)
    
    my_file = open(INPUT_FN + path, "r")
    sequences = my_file.read().split('\n')
    
    sequences_ = []
    
    y = []
    
    s = []
    
    for sequence in sequences:
        
        if 'class' in sequence:
            
            if s:
            
                sequences_.append(list(''.join(s)))
                        
            y.append(
                int(sequence.split('class:')[-1])
                )
            
            s = []
            
        else:
             
            s.append(sequence)
            
    sequences_.append(list(''.join(s)))

    loc = {'A':0,
           'C':1,
           'G':2,
           'T':3,
           'N':4,
           }
    
    X = np.zeros((len(sequences_), len(loc), len(sequences_[0])))
    
    for i in range(len(sequences_)):
        
        for j in range(len(sequences_[i])):
            
            X[i][loc[sequences_[i][j]]][j] = 1
    
    
    features = list(loc.keys())
    
    dtypes = [np.int16] * len(features)
    
    return X, y, features, dtypes, task, y_scale
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser('Main script for training ABR')

    parser.add_argument(
        '--dataname', 
        default='1_PARCLIP_AGO1234_hg19', 
        type=str
        )

    parser.add_argument(
        '--transformation', 
        default=None, 
        type=str
        )

    args = parser.parse_args()

    X, y, features, dtypes, task, y_scale = load_rna_binding(args.dataname)

    generator = Generator(
        X, 
        y,
        features, 
        dtypes, 
        task, 
        y_scale,
        args.transformation
        )
    
    with open(
            './data/preprocessed/generator_{}.pk'.format(args.dataname), 
            'wb') as fd:
        
        pickle.dump(
            generator, 
            fd, 
            protocol=4
            )