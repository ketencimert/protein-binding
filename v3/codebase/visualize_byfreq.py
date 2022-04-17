# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:52:22 2022

@author: Mert
"""

from collections import defaultdict
import numpy as np

import torch
from torch.distributions.normal import Normal

from torch.nn import BCELoss
from torch import nn
import torch.nn.functional as F

keys={
      0:"A",
      1:"C",
      2:"G",
      3:"T"
      }

counts = np.zeros((4,20))

index = 0

for (x_tr, y_tr) in train_dataloader:

    w, _ = self.qw_network()
    
    assignment = self.pz_wx_network(self, w, x_tr).max(-1)[1]
    
    #2. Get motif tensor
    
    x = x_tr[assignment == index,:,:]
    
    pwm = self.qw_network.loc.cpu()[index]
    w_z = pwm.view(4, -1)

    theta_x = self.py_xzw_network.theta_x(x)
    if self.py_xzw_network.use_z:
        theta_z = self.py_xzw_network.theta_z(torch.tensor([index]).to(x.device).repeat_interleave(x.size(0),0))
        theta = self.py_xzw_network.theta(torch.cat([theta_x, theta_z],-1))
    else:
        theta = theta_x

    y = (F.conv1d(x, w_z.unsqueeze(0)).squeeze(1)).squeeze(1).max(-1)[1]
    
    align = []
    
    for i in range(x.size(0)):
        align.append(x[i,:,y[i]:y[i]+20])
        
        for j in range(align[-1].size(1)):
        
            counts[align[-1][:,j].max(0)[1].item()][j] += 1

counts = np.log(counts / counts.sum(0)) - np.log(0.25)

pwm_df = pd.DataFrame(data = counts.transpose(), columns=("A","C","G","T"))
crp_logo = logomaker.Logo(pwm_df) # CCACCAGG(G/T)GGCG