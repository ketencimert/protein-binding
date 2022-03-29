# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:02:04 2022

@author: Mert
"""

import torch
from torch import nn

from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

class CNN(nn.Module):

    def __init__(self, 
                 n_output_channels = 1, 
                 filter_widths = [15, 5], 
                 num_chunks = 5, 
                 max_pool_factor = 4, 
                 nchannels = [4, 32, 32],
                 n_hidden = 32, 
                 dropout = 0.2,
                 num_motifs = 100,):
        
        super(CNN, self).__init__()
        self.rf = 0 # running estimate of the receptive field
        self.chunk_size = 1 # running estimate of num basepairs corresponding to one position after convolutions

        conv_layers = []
        for i in range(len(nchannels)-1):
            conv_layers += [ nn.Conv1d(nchannels[i], nchannels[i+1], filter_widths[i], padding = 0),
                        nn.BatchNorm1d(nchannels[i+1]), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                        nn.Dropout2d(dropout), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                        nn.MaxPool1d(max_pool_factor), 
                        nn.ELU(inplace=True)  ] # popular alternative to ReLU: https://arxiv.org/abs/1511.07289
            assert(filter_widths[i] % 2 == 1) # assume this
            self.rf += (filter_widths[i] - 1) * self.chunk_size
            self.chunk_size *= max_pool_factor

        # If you have a model with lots of layers, you can create a list first and 
        # then use the * operator to expand the list into positional arguments, like this:
        self.conv_net = nn.Sequential(*conv_layers)

        self.seq_len = num_chunks * self.chunk_size + self.rf # amount of sequence context required
        
        n_output_channels = num_motifs
        
        print("Receptive field:", self.rf, "Chunk size:", self.chunk_size, "Number chunks:", num_chunks)

        self.dense_net = nn.Sequential( nn.Linear(nchannels[-1] * num_chunks, n_hidden),
                                        nn.Dropout(dropout),
                                        nn.ELU(inplace=True), 
                                        nn.Linear(n_hidden, num_motifs) )

    def forward(self, x):
        net = self.conv_net(x)
        net = net.view(net.size(0), -1)
        net = self.dense_net(net)
        return(net)

class ANN(nn.Module):

    def __init__(self, 
                 embedding_dim=128,
                 nhead=16, #attention heads
                 nucleatoide_size=4, #number of base pairs,
                 encoder_layer=6,#number of consecutive attention networks
                 ):

        super(ANN, self).__init__()
        
        self.embedding_matrix =  nn.Parameter(
            torch.randn(
                nucleatoide_size,
                embedding_dim
                ),
            requires_grad = True
            )
        
        
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead
            )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=1
            )
        
    def forward(self, x):
        
        z = torch.matmul(torch.transpose(x, 2, 1), self.embedding_matrix)
        
        z = self.transformer_encoder(z) #torch.transpose(x, 2,1)
        
        return z.mean(1)


class FNN(nn.Module):

    def __init__(self,
                 dropout_input=0.5,
                 dropout_intermediate=0.5,
                 layer_size=[100],
                 activation='elu',
                 norm='layernorm'
                 ):

        super(FNN, self).__init__()

        self.loc_layers = []

        if activation == 'relu':

            activation = nn.ReLU()

        elif activation == 'tanh':

            activation = nn.Tanh()

        elif activation == 'elu':

            activation = nn.ELU()

        elif activation == 'selu':

            activation = nn.SELU()

        elif activation == 'silu':

            activation = nn.SiLU()

        self.loc_layers.append(nn.BatchNorm1d(layer_size[0]))

        self.loc_layers.append(nn.Dropout(dropout_input))

        for i in range(len(layer_size)-2):

            self.loc_layers.append(nn.Linear(layer_size[i], layer_size[i+1]))

            self.loc_layers.append(activation)

            self.loc_layers.append(nn.Dropout(dropout_intermediate))

        self.loc_layers.append(nn.Linear(layer_size[-2], layer_size[-1]))

        self.loc_layers = nn.Sequential(*self.loc_layers)

    def forward(self, x):

        loc = self.loc_layers(x)

        return loc
