# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:02:04 2022

@author: Mert
"""

import torch
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

from torch import nn
import torch.nn.functional as F

from networks import CNN, FFN

from utils import (
    kl_divergence_1,
    roc_auc,
    pr_auc
    )

"""
p*?_network class is a non-linear function approximator that
parameterizes a distribution of interest (in this case p*?)


"""
class py_xzw_network(nn.Module):
    def __init__(self,
                 filter_widths=[15, 5],
                 num_chunks=5,
                 max_pool_factor=4,
                 nchannels=[4, 32, 32],
                 n_hidden=32,
                 dropout=0.2,
                 num_motifs=100,
                 kernel_size=20,
                 stride=1,
                 use_z=True
                 ):
        super(py_xzw_network, self).__init__()

        self.use_z = use_z
        #vectorize x using CNN network
        self.theta_x = CNN(
            filter_widths=filter_widths,
            num_chunks=num_chunks,
            max_pool_factor=max_pool_factor,
            nchannels=nchannels,
            n_hidden=n_hidden,
            dropout=dropout,
            num_motifs=num_motifs,
            kernel_size=kernel_size,
            stride=stride,
            use_z=use_z,
            )

        self.seq_len = self.theta_x.seq_len
        self.output_dim = self.theta_x.output_dim
        #create an embedding layer for z
        if self.use_z:
            layer_size = [self.output_dim*2] + nchannels[1:] + [self.output_dim]
            self.theta_z = nn.Embedding(
                num_embeddings=num_motifs,
                embedding_dim=self.output_dim
                )
            self.theta = FFN(
                dropout=dropout,
                layer_size=layer_size,
                activation='elu',
                )

    def forward(self, x, z, w):
        alpha = 0.2
        #for soft convolution component
        theta_x = self.theta_x(x)
        if self.use_z:
            z = torch.tensor([z]).to(x.device)
            theta_z = self.theta_z(z).repeat_interleave(x.size(0), 0)
            theta = self.theta(torch.cat([theta_x, theta_z],-1))
        else:
            theta = theta_x

        w_z = w[z,:].view(4, -1)
        #add soft and hard convolution components
        y_ = F.conv1d(x, w_z.unsqueeze(0))
        y = alpha * torch.sum(y_.squeeze(1) * theta, -1)
        y += (1-alpha) * torch.max(y_, -1)[0].squeeze(1) #hard convolution component
        return y


def pz_wx_network(model, w, x):
    
    #Computes the posterior distribution of pz_wx given the model
    
    prior_loc, prior_scale = model.qw_network()

    pz_ywx = []

    for i in range(w.size(0)):

        py_xwz = 0

        for j in (0,1):

            py_xwz += model.py_xwz_dist(
                logits=model.py_xzw_network(x, i, w)
                ).log_prob(torch.Tensor([j]).to(x.device))

        pz_ywx.append(py_xwz.unsqueeze(-1))

    pz_ywx = torch.cat(pz_ywx, -1)
    pz_ywx = pz_ywx - pz_ywx.logsumexp(-1).unsqueeze(-1)
    pz_ywx = pz_ywx.exp()

    return pz_ywx

class qw_network(nn.Module):
    def __init__(self, num_motifs, kernel_size):
        super(qw_network, self).__init__()

        self.loc = nn.Parameter(
            torch.randn(num_motifs, 4*kernel_size),
            requires_grad=True,
            )

        self.scale = nn.Parameter(
            torch.randn(num_motifs, 4*kernel_size),
            requires_grad=True,
            )

    def forward(self):
        loc = self.loc
        scale = nn.Softplus()(self.scale)
        return loc, scale


class Model(nn.Module):

    def __init__(
            self,
            num_chunks=5,
            max_pool_factor=4,
            nchannels=[4, 32, 32],
            n_hidden=32,
            dropout=0.2,
            num_motifs=2,
            kernel_size=20,
            stride=1,
            use_z=True,
            metric='pr',
            ):
        super(Model, self).__init__()

        self.eps = 1e-40
        self.num_motifs = num_motifs

        #define distributions
        self.qw_dist = Normal
        self.py_xwz_dist = Bernoulli

        #parameterize distributions
        self.qw_network = qw_network(
            num_motifs=num_motifs,
            kernel_size=kernel_size,
            )

        self.py_xzw_network = py_xzw_network(
            num_chunks=num_chunks,
            max_pool_factor=max_pool_factor,
            nchannels=nchannels,
            n_hidden=n_hidden,
            dropout=dropout,
            num_motifs=num_motifs,
            kernel_size=kernel_size,
            stride=1,
            use_z=use_z
            )

        self.pz_wx_network = pz_wx_network

        #encourage sparsity for PWM matrix
        self.prior_loc = 0
        self.prior_scale = 1

        # self.prior_scale = 1e-2
        # self.prior_loc = -1e3*self.prior_scale 

        self.seq_len = self.py_xzw_network.seq_len
        #choose metric

        if metric == 'pr':
            self.metric = pr_auc
        else:
            self.metric = roc_auc

    def forward(self, x, y):

        #0. Initiate elbo
        elbo = 0

        #1. Compute posterior w
        posterior_loc, posterior_scale = self.qw_network()

        w = self.qw_dist(
            loc=posterior_loc,
            scale=posterior_scale
            ).rsample()

        #2. Categorical entropy
        z = self.pz_wx_network(self, w, x)

        elbo += (z * (self.eps + z).log()).sum(1)

        elbo += kl_divergence_1(
            posterior_loc,
            posterior_scale,
            self.prior_loc,
            self.prior_scale
            )

        #3.Compute the KLD - motif tensor
        posterior_loc, posterior_scale = self.qw_network()

        w = self.qw_dist(
            loc=posterior_loc,
            scale=posterior_scale
            ).rsample()

        elbo += kl_divergence_1(
            posterior_loc,
            posterior_scale,
            self.prior_loc,
            self.prior_scale
            )

        #4. Take the expectation w.r.t. motif assignment:
        for i in range(self.num_motifs):

            logits = self.py_xzw_network(x, i, w)
            
            if y is not None:
            
                generative_loglikelihood = self.py_xwz_dist(
                    logits=logits
                    ).log_prob(y)
    
                elbo -= 1e8 * z[:,i] * generative_loglikelihood

        loss = elbo.mean()

        y_pred = self.predict(x)
        score = self.metric(y_pred, y)

        return loss, score, y_pred

    def predict(self, x):

        #0. Need to compute:
        #E_{q(w),p(z|..)p(y|..)}[y] = E_{q(w),p(z|..)}[p(y|..)]

        #1. Get motif assignments:
        #We won't sample w - just use w_mean to approximate
        #for computational efficiency

        with torch.no_grad():

            w, _ = self.qw_network()

            z = self.pz_wx_network(self, w, x)

            #2. Now we marginalize E_{p(z|..)}[p(y|..)]:
            y_pred = torch.stack(
                [
                z[:,i] * torch.sigmoid(
                    self.py_xzw_network(x, i, w)
                    ) for i in range(z.size(1))
                ]
                ).sum(0)

        return y_pred