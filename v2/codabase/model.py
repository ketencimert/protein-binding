

import torch
from torch.distributions.normal import Normal

from torch.nn import BCELoss
from torch import nn
import torch.nn.functional as F

from codebase.networks import CNN

from codebase.utils import (
    kl_divergence_1,
    roc_auc,
    pr_auc
    )


class py_xw_network(nn.Module):
    def __init__(self):
        super(py_xw_network, self).__init__()
        
        #cache kernel size
        
    def forward(self, x, w_z):
        
        #compute the convolution
        
        w_z = w_z.view(4, -1)
            
        y = nn.Sigmoid()(
            F.conv1d(x, w_z.unsqueeze(0)).squeeze(1).max(-1)[0]
            )
        
        return y #gives the logit


class qz_x_network(nn.Module):
    def __init__(self, num_motifs):
        super(qz_x_network, self).__init__()
        
        #the parameters of this network is independent of PSSM/PWM
        
        self.inference_network = CNN(num_motifs=num_motifs)
        
    def forward(self, x):
        
        #batch_size x motif assignment probability
        
        z = nn.Softmax(-1)(self.inference_network(x))
    
        return z


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
        
        self.qw_z_dist = Normal
        self.pw_dist = Normal
        
        #parameterize distributions
        
        self.qz_x_network = qz_x_network(
            num_motifs=num_motifs
            )
        
        self.qw_network = qw_network(
            num_motifs=num_motifs,
            kernel_size=kernel_size,
            )
        
        self.py_xw_network = py_xw_network()
        
        #encourage sparsity for PWM matrix
        
        self.prior_scale = 1e-1
        self.prior_loc = -1e1*self.prior_scale 
        
        #encourage good predictions
        
        generative_scale = -15.
        self.generative_scale = nn.Parameter(
            torch.tensor([generative_scale]),
            requires_grad=True,
            )

        self.seq_len = self.qz_x_network.inference_network.seq_len

        #choose metric
        
        if metric == 'pr':
            self.metric = pr_auc
        else:
            self.metric = roc_auc

    def forward(self, x, y):
        
        #0. Initiate elbo
        elbo = 0

        #1.Compute variational entropy
        z = self.qz_x_network(x)
        elbo += (z * z.log()).sum(1) #for each x make a motif assignment
        
        #2.Compute the KLD of motif tensor
        posterior_loc, posterior_scale = self.qw_network()
        
        w = self.qw_z_dist(
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
            
            generative_loc = self.py_xw_network(x, w[i,:])
            
            generative_scale = 1 / nn.Softplus()(self.generative_scale)
            
            eps = 1e-16
            
            generative_loglikelihood = -torch.log(
                eps + (eps + generative_loc).pow(generative_scale) \
                    + (eps + 1-generative_loc).pow(generative_scale)
                )
    
            generative_loglikelihood -= generative_scale * BCELoss()(
                generative_loc,
                y
                )

            elbo -= z[:,i] * generative_loglikelihood
        
        loss = elbo.mean()
        
        y_pred = self.predict(x)
        
        score = self.metric(y_pred, y)
        
        return loss, score, y_pred

    def predict(self, x):
        
        #1. Get motif assignments
        motif_assignments = self.qz_x_network(x).max(-1)[1]
        
        #2. Get motif tensor
        posterior_loc, _ = self.qw_network()
        
        w_z = posterior_loc[motif_assignments, :]
        
        w_z = w_z.view(x.size(0), 4, -1)
        
        y = []
        
        for i in range(x.size(0)):
                
            y.append(
                F.conv1d(x[i,:,:].unsqueeze(0),
                         w_z[i,:,:].unsqueeze(0)).max(-1)[0]
                )
                
        y_pred = nn.Sigmoid()(torch.cat(y).view(-1))
        
        return y_pred
    
