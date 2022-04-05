
import os

import numpy as np

import torch

from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score


def kl_divergence_1(posterior_loc, posterior_scale, prior_loc, prior_scale):

    d = posterior_loc.size(-1)

    kld = 0.5 * (
        prior_scale**(-2) * posterior_scale.pow(2) \
            + prior_scale**(-2) * (prior_loc - posterior_loc).pow(2) \
                - d + 2*(np.log(prior_scale)-posterior_scale.log())
        ).sum(-1)

    return kld.sum()

def kl_divergence_2(posterior_loc, posterior_scale):

    d = posterior_loc.size(-1)

    kld = 0.5 * (
        posterior_scale.pow(2) + posterior_loc.pow(2)\
            - d - 2*posterior_scale.log()
        ).sum(-1)

    return kld.mean()

def kl_divergence_3(posterior_loc, posterior_cov, prior_loc, prior_scale):
    
    n, d = posterior_loc.size()
    
    prior_cov = prior_scale**2
        
    posterior_det = 2 * torch.sum(
        torch.diagonal(posterior_cov, dim1=2).log()
        )
    
    posterior_trace = torch.sum(
        (prior_cov**-1) * torch.diagonal(posterior_cov, dim1=2)
        )
    
    kld = (
        - posterior_det - d + posterior_trace + prior_cov**(-1) * (
            prior_loc - posterior_loc
            ).pow(2)
        ).sum(-1)
    
    return kld.sum()
    
    
def init_weights(layer):

    size = layer.weight.size()

    fan_out = size[0]

    fan_in = size[1]

    std = np.sqrt(2.0/(fan_in + fan_out))

    layer.weight.data.normal_(0.0, std)

    layer.bias.data.normal_(0.0, 0.001)

    return layer

def mape(y_pred,y_va):

    diff = torch.abs(y_va - y_pred)/y_va

    return diff.mean()

def rmse(y_pred,y_va):

    diff = y_pred - y_va

    diff = diff*diff

    diff = diff.mean()

    return diff**0.5

def accuracy(y_pred,y_va):

    return ((y_pred>0.5)==y_va).sum()/y_va.shape[0]

def roc_auc(y_pred, y_va):

    try:

        y_pred = y_pred.cpu().detach().numpy()

        y_va = y_va.cpu().detach().numpy()

    except:

        pass

    try:

        res = roc_auc_score(y_va.astype(int), y_pred)

    except:

        res = np.asarray([np.nan])

    return res

def pr_auc(y_pred, y_va):

    try:

        y_pred = y_pred.cpu().detach().numpy()

        y_va = y_va.cpu().detach().numpy()

    except:

        pass

    try:

        res = average_precision_score(y_va, y_pred)

    except:

        res = np.asarray([np.nan])

    return res

def save(best_model, data):

    path = './saves/'

    os.makedirs(path, mode=0o777, exist_ok=True)

    torch.save(best_model, path+'best_model_{}.pth'.format(data))

def save_runs(args, seed, fold_scores, path='./runs_/v2'):

    os.makedirs(path, mode=0o777, exist_ok=True)

    experiment = 0

    for folders in next(os.walk(path))[1]:

        if 'experiment' in folders:

            experiment = max(int(folders.split('experiment')[-1]), experiment)

    experiment += 1

    experiment_path = path+'/experiment{}'.format(experiment)

    os.makedirs(
        experiment_path,
        mode=0o777,
        exist_ok=False
        )

    results_path = experiment_path+'/results.txt'

    file = open(results_path, "w")

    file.write("\n args : {}".format(vars(args)))

    file.write("\n Seed : {}".format(seed))

    file.write("\n\n Mean model Fold Score : {}".format(np.mean(fold_scores)))

    file.write("\n\n STDEV model Fold Scores : {}".format(
        np.std(fold_scores)/np.sqrt(len(fold_scores))))


    file.write("\n\n Worst model Fold Score : {}".format(min(fold_scores)))

    file.write("\n\n Best model Fold Score : {}".format(max(fold_scores)))

    fold_scores = [str(x) for x in fold_scores]

    file.write("\n\n Fold Scores : {}".format(
        ','.join(fold_scores))
        )

    file.close()
