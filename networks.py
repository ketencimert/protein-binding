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
                 filter_widths = [15, 5],
                 num_chunks = 5,
                 max_pool_factor = 4,
                 nchannels = [4, 32, 32],
                 n_hidden = 32,
                 dropout = 0.2,
                 num_motifs=100,
                 kernel_size=20,
                 stride=1,
                 use_z=True
                 ):

        super(CNN, self).__init__()
        self.rf = 0 # running estimate of the receptive field
        self.chunk_size = 1 # running estimate of num basepairs corresponding to one position after convolutions

        conv_layers = []
        for i in range(len(nchannels)-1):
            conv_layers += [
                nn.Conv1d(nchannels[i], nchannels[i+1], filter_widths[i], padding = 0),
                nn.BatchNorm1d(nchannels[i+1]), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                nn.Dropout2d(dropout), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                nn.MaxPool1d(max_pool_factor),
                nn.ELU(inplace=True)
                ] # popular alternative to ReLU: https://arxiv.org/abs/1511.07289

            assert(filter_widths[i] % 2 == 1) # assume this
            self.rf += (filter_widths[i] - 1) * self.chunk_size
            self.chunk_size *= max_pool_factor

        # If you have a model with lots of layers, you can create a list first and
        # then use the * operator to expand the list into positional arguments, like this:
        self.conv_net = nn.Sequential(*conv_layers)

        self.seq_len = num_chunks * self.chunk_size + self.rf # amount of sequence context required

        print("Receptive field:", self.rf, "Chunk size:", self.chunk_size, "Number chunks:", num_chunks)

        self.output_dim = int((self.seq_len - kernel_size) / stride + 1)

        dense_net = [
            nn.Linear(nchannels[-1] * num_chunks, n_hidden),
            nn.Dropout(dropout),
            nn.ELU(inplace=True),
            nn.Linear(n_hidden, self.output_dim)
            ]

        if not use_z:

            dense_net.append(nn.Softmax(-1))

        self.dense_net = nn.Sequential(*dense_net)

    def forward(self, x):
        net = self.conv_net(x)
        net = net.view(net.size(0), -1)
        net = self.dense_net(net)
        return(net)


class FFN(nn.Module):

    def __init__(self,
                 dropout=0.2,
                 layer_size=[1,100],
                 activation='elu',
                 ):

        super(FFN, self).__init__()

        layers = []

        activation = {
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'elu':nn.ELU(),
            'selu':nn.SELU(),
            'silu':nn.SiLU(),
            }[activation]

        for i in range(len(layer_size)-2):

            layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            layers.append(nn.BatchNorm1d(layer_size[i+1]))
            layers.append(nn.Dropout(dropout))
            layers.append(activation)

        layers.append(nn.Linear(layer_size[-2], layer_size[-1]))
        layers.append(nn.Softmax(-1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        loc = self.layers(x)
        return loc

###############################################################################
#                              BASELINE NETWORKS                              #
###############################################################################

class CNN_BASELINE(nn.Module):

    def __init__(self,
                 n_output_channels = 1,
                 filter_widths = [15, 5],
                 num_chunks = 5,
                 max_pool_factor = 4,
                 nchannels = [4, 32, 32],
                 n_hidden = 32,
                 dropout = 0.2):

        super(CNN_BASELINE, self).__init__()
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

        print("Receptive field:", self.rf, "Chunk size:", self.chunk_size, "Number chunks:", num_chunks)

        self.dense_net = nn.Sequential( nn.Linear(nchannels[-1] * num_chunks, n_hidden),
                                        nn.Dropout(dropout),
                                        nn.ELU(inplace=True),
                                        nn.Linear(n_hidden, n_output_channels) )
        self.output_len = n_output_channels
    def forward(self, x):
        net = self.conv_net(x)
        net = net.view(net.size(0), -1)
        net = self.dense_net(net)
        return(net)


class TN_BASELINE(nn.Module):

    def __init__(self,
                 embedding_dim=128,
                 nhead=8, #attention heads
                 nucleatoide_size=4, #number of base pairs,
                 encoder_layer=3,#number of consecutive attention networks
                 output_dim=1,
                 ):

        super(TN_BASELINE, self).__init__()

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

        self.sequential = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                        nn.ReLU(),
                                        nn.Linear(embedding_dim, output_dim)
                                        )


    def forward(self, x):

        z = torch.matmul(torch.transpose(x, 2, 1), self.embedding_matrix)

        z = self.transformer_encoder(z) #torch.transpose(x, 2,1)

        return self.sequential(z.mean(1))


class FFNN_BASELINE(nn.Module):

    def __init__(self,
                  dropout=0.5,
                  layer_size=[440, 64, 64, 32, 1],
                  activation='elu',
                  ):

        super(FFNN_BASELINE, self).__init__()

        self.layers = []

        activation = {
            'relu':nn.ReLU(),
            'tanh':nn.Tanh(),
            'elu':nn.ELU(),
            'selu':nn.SELU(),
            'silu':nn.SiLU(),
            }[activation]

        self.layers.append(nn.BatchNorm1d(layer_size[0]))

        for i in range(len(layer_size)-2):

            self.layers.append(nn.Linear(layer_size[i], layer_size[i+1]))
            self.layers.append(activation)
            self.layers.append(nn.LayerNorm(layer_size[i+1]))
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(layer_size[-2], layer_size[-1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):

        loc = self.layers(x.view(x.size(0), -1))

        return loc


class TNCNN_BASELINE(nn.Module):

    def __init__(self,
                 n_output_channels = 100,
                 filter_widths = [15, 5],
                 num_chunks = 5,
                 max_pool_factor = 4,
                 nchannels = [4, 32, 32],
                 n_hidden = 32,
                 dropout = 0.2,
                 embedding_dim=36,
                 nhead=1, #attention heads
                 nucleatoide_size=4, #number of base pairs,
                 encoder_layer=6, #number of consecutive attention networks,
                 hidden_layers=[32, 32],
                 layer_size=[100],
                 activation='elu',
                 norm='layernorm'
                 ):

        super(TNCNN_BASELINE, self).__init__()

        self.cnn = CNN_BASELINE(
                 n_output_channels=n_output_channels,
                 filter_widths=filter_widths,
                 num_chunks=num_chunks,
                 max_pool_factor=max_pool_factor,
                 nchannels=nchannels,
                 n_hidden=n_hidden,
                 dropout=dropout,
                 )

        self.tn = TN_BASELINE(
                 embedding_dim=embedding_dim,
                 nhead=nhead,
                 nucleatoide_size=nucleatoide_size,
                 encoder_layer=encoder_layer,
                 output_dim=n_output_channels,
                 )

        self.seq_len = self.cnn.seq_len

        output_len = self.cnn.output_len*2

        layer_size = [output_len] + hidden_layers + [1]

        self.ffnn = FFNN_BASELINE(
                 dropout=dropout,
                 layer_size=layer_size,
                 activation=activation,
            )

    def forward(self, x):

        z1 = self.cnn(x)
        z2 = self.tn(x)

        z = torch.cat([z1, z2], 1)

        y = self.ffnn(z)
        return y
