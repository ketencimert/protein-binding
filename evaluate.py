# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 20:50:56 2022

@author: Mert
"""

import argparse
import random

import numpy as np

import torch

from codebase.model import Model
from codebase.networks import (CNN_BASELINE,
                               TN_BASELINE,
                               FFNN_BASELINE,
                               TNCNN_BASELINE
                               )

from codebase.utils import dataloader
from codebase.utils import pr_auc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #path args
    parser.add_argument('--datadir', default='./data', type=str)
    parser.add_argument('--dataname', default='atac', type=str)

    #device args
    parser.add_argument('--device', default='cpu', type=str)

    #optimization / batch args
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    #model args
    parser.add_argument('--model', default='model', type=str)

    args = parser.parse_args()
    print(args)

    seed = 12345
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    modelname = args.model

    with torch.no_grad():

        if modelname in ['ffnn', 'cnn', 'tn', 'tncnn']:

            model = {'ffnn':FFNN_BASELINE(),
                     'cnn':CNN_BASELINE(),
                     'tn':TN_BASELINE(),
                     'tncnn':TNCNN_BASELINE(),
                     }[args.model].to(args.device)

            model.load_state_dict(
                torch.load('./saves/{}_{}_checkpoint.pt'.format(modelname, args.dataname),
                           map_location=args.device
                           )
                )

            model.eval()

            if args.model not in ['cnn', 'tncnn']:
                #if not cnn based model use the standard seq len at assignment2 = 110
                model.seq_len = 110
            else:
                seq_len = None

            _, _, test_dataloader = dataloader(
              datadir=args.datadir,
              dataname=args.dataname,
              seq_len=model.seq_len,
              batch_size=args.batch_size,
              num_workers=args.num_workers,
              test=True,
              )

            if args.dataname == 'chip':

                scores = []
                for (x_te, y_te) in test_dataloader:

                    x_te, y_te = x_te.to(args.device), y_te.to(args.device)
                    score = pr_auc(model(x_te), y_te)
                    scores.append(score.item())

                print('Test PR-AUC is {}'.format(np.mean(scores)))

            else:

                outputs = []
                for x in test_dataloader: # iterate over batches
                    x = x.to(args.device)
                    output = model(x).squeeze() # your awesome model here!
                    output = torch.sigmoid(output)
                    output_np = output.detach().cpu().numpy()
                    outputs.append(output_np)
                output_np = np.concatenate(outputs)
                np.save('./saves/{}_{}_predictions'.format(
                    modelname, args.dataname
                    ),
                    output_np
                    )

        else:

            model = torch.load(
                './saves/model_{}_checkpoint.pth'.format(args.dataname)
                ).to(args.device)

            model.eval()

            _, _, test_dataloader = dataloader(
            datadir=args.datadir,
            dataname=args.dataname,
            seq_len=model.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            test=True,
            )

            if args.dataname == 'chip':

                scores = []
                for (x_te, y_te) in test_dataloader:

                    x_te, y_te = x_te.to(args.device), y_te.to(args.device)
                    elbo, score, y_pred = model(x_te, y_te)
                    scores.append(score.item())

                print('Test PR-AUC is {}'.format(np.mean(scores)))

            else:

                outputs = []
                for x in test_dataloader: # iterate over batches
                    x = x.to(args.device)
                    _, _, output = model(x, None) # your awesome model here!
                    output_np = output.detach().cpu().numpy()
                    outputs.append(output_np)
                output_np = np.concatenate(outputs)
                np.save('./saves/{}_{}_predictions'.format(
                    modelname, args.dataname
                    ),
                    output_np
                    )
