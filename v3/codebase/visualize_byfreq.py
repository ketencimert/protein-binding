# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:52:22 2022

@author: Mert
"""

from collections import defaultdict

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
    
    assignment = self.pz_wyx_network(self, w,y_tr, x_tr).max(-1)[1]
    
    #2. Get motif tensor
    
    x = x_tr[assignment,:,:]
    
    pwm = self.qw_network.loc.cpu()[index]
    w_z = pwm.view(4, -1)
        
    y = F.conv1d(x, w_z.unsqueeze(0)).squeeze(1).max(-1)[1]
    
    align = []
    
    for i in range(x.size(0)):
        align.append(x[i,:,y[i]:y[i]+20])
        
        for j in range(align[-1].size(1)):
        
            counts[align[-1][:,j].max(0)[1].item()][j] += 1

counts = np.log(counts / counts.sum(0)) - np.log(0.25)

pwm_df = pd.DataFrame(data = counts.transpose(), columns=("A","C","G","T"))
crp_logo = logomaker.Logo(pwm_df) # CCACCAGG(G/T)GGCG