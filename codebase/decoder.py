# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 21:55:09 2022

@author: Mert
"""

import torch
from torch import nn
import torch.nn.functional as F

class GenerativeNetwork(nn.Module):

    def __init__(self,
                 amortize_bias,
                 dropout_output
                 ):

        super(GenerativeNetwork, self).__init__()

    def forward(self, x, pwm):
        
        pwm = pwm.view(x.size(0), 4, 20)
        y = []
        
        for i in range(x.size(0)):
        
            y.append(
                F.conv1d(x[i,:,:].unsqueeze(0),
                         pwm[i,:,:].unsqueeze(0)).max(-1)[0]
                )
        
        # y = torch.sum(x * pwm, -1).sum(-1)
        y = torch.cat(y).view(-1)
        return y #gives the logit