
import torch
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

from torch.nn import BCELoss
from torch import nn
import torch.nn.functional as F

from codebase.networks import CNN, FFN

from codebase.utils import (
    kl_divergence_1,
    roc_auc,
    pr_auc
    )


class py_xwz_network(nn.Module):
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
                 use_z=True):
        super(py_xwz_network, self).__init__()

        #cache kernel size
        self.use_z = use_z
        
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

        #compute the convolution
        
        theta_x = self.theta_x(x)
        if self.use_z:
            z = torch.tensor([z]).to(x.device)
            theta_z = self.theta_z(z).repeat_interleave(x.size(0), 0)
            theta = self.theta(torch.cat([theta_x, theta_z],-1))
        else:
            theta = theta_x
        
        w_z = w[z,:].view(4, -1)

        y = nn.Sigmoid()(
            ((F.conv1d(x, w_z.unsqueeze(0))).squeeze(1) * theta).sum(-1)
            )

        return y #gives the logit


def pz_wx_network(model, w, x):

        prior_loc, prior_scale = model.pw_network()

        pz_ywx = []

        for i in range(w.size(0)):

            # pw = Normal(
            #     loc=prior_loc[i],
            #     scale=prior_scale[i]
            #     ).log_prob(w[i]).sum(-1)
            py_xwz = 0

            for j in (0,1):

                py_xwz += Bernoulli(
                    probs=model.py_xwz_network(x, i, w)
                    ).log_prob(torch.Tensor([j]).to(x.device))

                # generative_loc = model.py_xwz_network(x, w[i])

                # generative_scale = 1/nn.Softplus()(model.generative_scale)

                # eps = 1e-16

                # generative_loglikelihood = -torch.log(
                # eps + (eps + generative_loc).pow(generative_scale) \
                #     + (eps + 1-generative_loc).pow(generative_scale)
                # )

                # generative_loglikelihood -= generative_scale * BCELoss(reduction='none')(
                # generative_loc,
                # torch.Tensor([j]).to(x.device).repeat_interleave(generative_loc.size(0),0)
                # )

                # py_xwz += generative_loglikelihood

            pz_ywx.append(py_xwz.unsqueeze(-1))

        pz_ywx = torch.cat(pz_ywx, -1)

        pz_ywx = pz_ywx - pz_ywx.logsumexp(-1).unsqueeze(-1)

        pz_ywx = pz_ywx.exp()

        return pz_ywx


class pw_network(nn.Module):
    def __init__(self, num_motifs, kernel_size):
        super(pw_network, self).__init__()

        self.loc = nn.Parameter(
            torch.randn(num_motifs, 4*kernel_size),
            requires_grad=True,
            )

        self.scale = nn.Parameter(
            torch.randn(num_motifs, 4*kernel_size),
            requires_grad=True,
            )

    def forward(self):

        #pick ith motif

        loc = self.loc

        scale = nn.Softplus()(self.scale)

        return loc, scale


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

        #pick ith motif

        loc = self.loc

        scale = nn.Softplus()(self.scale)

        return loc, scale


class Model(nn.Module):

    def __init__(
            self,
            num_motifs=2,
            metric='pr',
            kernel_size=20,
            ):

        super(Model, self).__init__()

        self.num_motifs = num_motifs

        #choose prior and posterior distributions

        self.qw_dist = Normal
        self.pw_dist = Normal

        #parameterize distributions

        self.qw_network = qw_network(
            num_motifs=num_motifs,
            kernel_size=kernel_size,
            )


        self.py_xwz_network = py_xwz_network()

        self.pw_network = pw_network(
            num_motifs=num_motifs,
            kernel_size=kernel_size,
            )

        self.pz_wx_network = pz_wx_network


        #encourage sparsity for PWM matrix

        self.prior_scale = 1e-3
        self.prior_loc = -1e-2

        #encourage good predictions

        generative_scale = -15.
        self.generative_scale = nn.Parameter(
            torch.tensor([generative_scale]),
            requires_grad=True,
            )

        self.seq_len = 110
        #choose metric

        if metric == 'pr':
            self.metric = pr_auc
        else:
            self.metric = roc_auc

    def forward(self, x, y):

        #0. Initiate elbo
        elbo = 0

        posterior_loc, posterior_scale = self.qw_network()

        w = self.qw_dist(
            loc=posterior_loc,
            scale=posterior_scale
            ).rsample()

        z = self.pz_wx_network(self, w, x)

        elbo += (z * (1e-20 + z).log()).sum(1) #for each x make a motif assignment

        elbo += kl_divergence_1(
            posterior_loc,
            posterior_scale,
            self.prior_loc,
            self.prior_scale
            )

        #2.Compute the KLD of motif tensor
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

        #2. Take the expectation w.r.t. motif assignment:
        for i in range(self.num_motifs):

            generative_loc = self.py_xwz_network(x, i, w)

            generative_scale = 1/nn.Softplus()(self.generative_scale)

            eps = 1e-16

            generative_loglikelihood = -torch.log(
                eps + (eps + generative_loc).pow(generative_scale) \
                    + (eps + 1-generative_loc).pow(generative_scale)
                )
            # generative_loglikelihood = Bernoulli(probs=generative_loc).log_prob(y)

            generative_loglikelihood -= generative_scale * BCELoss()(
                generative_loc,
                y
                )

            elbo -= z[:,i] * generative_loglikelihood

        loss = elbo.mean()

        y_pred = self.predict(x, y)

        score = self.metric(y_pred, y)

        return loss, score, y_pred

    def predict(self, x, y):
        
        #E_{q(w),p(z|..)p(y|..)}[y] = E_{q(w),p(z|..)}[p(y|..)]
        
        #1. Get motif assignments
        #We won't sample w - just use w_mean to approximate 
        #for computational efficiency
        w, _ = self.qw_network()
        
        z = self.pz_wx_network(self, w, x)
        
        #2. Now we marginalize E_{p(z|..)}[p(y|..)]
        y_pred = torch.stack(
            [z[:,i] * self.py_xwz_network(x, i, w) for i in range(z.size(1))]
            ).sum(0)
            
        
        
        # motif_assignments = self.pz_wyx_network(self,w,y,x).max(-1)[1]

        # #2. Get motif tensor
        # posterior_loc = w

        # w_z = posterior_loc[motif_assignments, :]

        # w_z = w_z.view(x.size(0), 4, -1)

        # y = []
        
        # theta = self.py_xwz_network.theta(x)
        
        # for i in range(x.size(0)):
            
        #     y.append((
        #         F.conv1d(x[i,:,:].unsqueeze(0),
        #                  w_z[i,:,:].unsqueeze(0)) * theta[i,:]).sum()
        #         )

        # y_pred = nn.Sigmoid()(torch.stack(y).view(-1))

        return y_pred
