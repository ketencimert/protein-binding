# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 23:31:21 2022

@author: Mert
"""

import argparse

import logomaker
import torch

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import torch.nn.functional as F

from utils import dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #path args
    parser.add_argument('--datadir', default='./data', type=str)
    parser.add_argument('--dataname', default='atac', type=str)

    #device args
    parser.add_argument('--device', default='cuda', type=str)

    #optimization / batch args
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    args = parser.parse_args()

    counts_ = np.zeros((4,15))

    index = 0 #23 28
    
    with torch.no_grad():
    
        best_model = torch.load(
            'saves/model_atac_nummotifs_30_checkpoint.pth'.format(args.dataname)
            ).to(args.device)
        
        best_model.eval()
        
        pwm = best_model.qw_network.loc.cpu()[index].view(4,15).cpu().detach().numpy()
        # pwm = pwm / pwm.sum()
        pwm_df = pd.DataFrame(data = pwm.transpose(), columns=("A","C","G","T"))
        crp_logo = logomaker.Logo(pwm_df) # CCACCAGG(G/T)GGCG
    
        train_dataloader, validation_dataloader, _ = dataloader(
            datadir=args.datadir,
            dataname=args.dataname,
            seq_len=best_model.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            )
        
        align = []
        
        for (x_tr, y_tr) in train_dataloader:
            
            x_tr, y_tr = x_tr.to(args.device), y_tr.to(args.device)
            
            posterior_loc, posterior_scale = best_model.qw_network()
            w = posterior_loc
            # w = best_model.qw_dist(
            #     loc=posterior_loc,
            #     scale=posterior_scale
            #     ).rsample()
            
            assignment = best_model.pz_wx_network(best_model, w, x_tr).max(-1)[1]
            
            #2. Get motif tensor
            try:
                x = x_tr[assignment == index,:,:]
                
                pwm = best_model.qw_network.loc[index]
                w_z = pwm.view(4, -1)
            
                theta_x = best_model.py_xzw_network.theta_x(x)
                if best_model.py_xzw_network.use_z:
                    theta_z = best_model.py_xzw_network.theta_z(
                        torch.tensor([index]).to(x.device).repeat_interleave(x.size(0),0)
                        )
                    theta = best_model.py_xzw_network.theta(torch.cat([theta_x, theta_z],-1))
                else:
                    theta = theta_x
            
                # y = (F.conv1d(x, w_z.unsqueeze(0)).squeeze(1)).squeeze(1).max(-1)[1] #maybe use model.fowrad?
        
                y_ = F.conv1d(x, w_z.unsqueeze(0))
                y = 0.2 * y_.squeeze(1) * theta
                y += 0.8 *y_.squeeze(1)
                y = y.max(-1)[1]
                
                
                for i in range(x.size(0)):
                    align.append(x[i,:,y[i]:y[i]+15])
                    
                    for j in range(align[-1].size(1)):
                    
                        counts_[align[-1][:,j].max(0)[1].item()][j] += 1
            except:
                pass
        counts_ += 1e-1
        counts = np.log(counts_ / counts_.sum(0)) - np.log(0.25)
        
        pwm_df = pd.DataFrame(data = counts.transpose(), columns=("A","C","G","T"))
        
        # create Logo object
        crp_logo = logomaker.Logo(pwm_df,
                                  shade_below=.5,
                                  fade_below=.5,
                                  font_name='Arial Rounded MT Bold',
                                  )
        
        # style using Logo methods
        crp_logo.style_spines(visible=False)
        crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
        crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)
        
        # style using Axes methods
        crp_logo.ax.set_title('Motif Assignment / Cluster: {}'.format(index))
        crp_logo.ax.xaxis.set_ticks_position('none')
        crp_logo.ax.xaxis.set_tick_params(pad=-1)

    
    #GCCCCCTGGTGGC