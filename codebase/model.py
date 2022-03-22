import numpy as np

import torch
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace

from torch.nn import BCELoss
from torch import nn

from codebase.cnn_1d import CNN_1d
from codebase.decoder import GenerativeNetwork
from codebase.encoder import VariationalNetwork

from codebase.utils import (
    kl_divergence_1,
    kl_divergence_2,
    rmse,
    accuracy,
    roc_auc,
    pr_auc
    )


class ConvAVBR(nn.Module):

    def __init__(self,
                 n_output_channels = 1,
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
                 dropout_input=0.5,
                 dropout_intermediate=0.5,
                 layer_size=[100],
                 activation='elu',
                 norm='layernorm',
                 amortize_bias=False,
                 task='classification',
                 auc='pr',
                 dropout_output=0
                 ):

        super(ConvAVBR, self).__init__()

        self.prior_dist = Normal

        if not amortize_bias:

            layer_size[-1] -= 1

        if task == 'regression':

            self.metric = rmse

        else:

            if auc == 'pr':

                self.metric = pr_auc

            else:

                self.metric = roc_auc

        self.task = task

        self.auc = auc

        self.amortize_bias = amortize_bias

        self.prior_scale = 5e-3

        self.prior_loc = -self.prior_scale * 10

        self.loc_encoder = CNN_1d(
                  )

        self.scale_encoder = CNN_1d(
                  )

        self.decoder = GenerativeNetwork(
            amortize_bias=amortize_bias,
            dropout_output=dropout_output,
            )

        generative_scale = -10.

        self.generative_scale = nn.Parameter(
            torch.tensor([generative_scale]),
            requires_grad=True,
            )

        self.seq_len = self.loc_encoder.seq_len

    def forward(self, x, y):

        generative_scale = nn.Softplus()(self.generative_scale)

        prior_loglikelihood = 0

        posterior_loc = self.loc_encoder(x)
        
        posterior_scale = nn.Softplus()(self.scale_encoder(x))

        w = Normal(loc=posterior_loc, scale=posterior_scale).rsample()

        generative_loc = self.decoder(x, w)

        if self.prior_dist is Normal:

            kl = kl_divergence_1(
                posterior_loc,
                posterior_scale,
                self.prior_loc,
                self.prior_scale
                )

        else:

            kl = Normal(
                loc=posterior_loc,
                scale=posterior_scale
                ).log_prob(w).sum(-1).mean()

            kl -= self.prior_dist(
                loc=self.prior_loc,
                scale=self.prior_scale
                ).log_prob(w).sum(-1).mean()

        generative_scale = 1/generative_scale

        generative_loc_ = nn.Sigmoid()(generative_loc)

        eps = 1e-16

        generative_loglikelihood = -torch.log(
            eps + (eps + generative_loc_).pow(generative_scale) \
                + (eps + 1-generative_loc_).pow(generative_scale)
            )

        generative_loglikelihood -= generative_scale * BCELoss()(
            generative_loc_,
            y
            )

        generative_loglikelihood = generative_loglikelihood.mean()

        # k = 1 + np.pi * (
        #     generative_scale
        #     )**2 * torch.sum(
        #         x.pow(2).view(w.size()) * posterior_scale.pow(2),
        #         -1
        #         )/8

        # k = generative_scale * k.pow(-0.5)

        generative_loc = nn.Sigmoid()(generative_loc)

        score = self.metric(generative_loc, y)

        loss = kl - generative_loglikelihood - prior_loglikelihood

        if not self.training:

            loss = loss.item()

            score = score.item()

        return loss, score, generative_loc
