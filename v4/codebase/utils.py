
import pickle

import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


bases={'A':0, 'C':1, 'G':2, 'T':3 }


def one_hot(string):

    res = np.zeros((4, len(string)),
                   dtype=np.float32)

    for j in range(len(string)):
        if string[j] in bases: # bases can be 'N' signifying missing: this corresponds to all 0 in the encoding
            res[ bases[ string[j] ], j ]= 1.

    return res


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


class BedPeaksDataset(torch.utils.data.IterableDataset):

    def __init__(self, atac_data, genome, context_length):
        super(BedPeaksDataset, self).__init__()
        self.context_length = context_length
        self.atac_data = atac_data
        self.genome = genome

    def __iter__(self):
        prev_end = 0
        prev_chrom = ""
        for i,row in enumerate(self.atac_data.itertuples()):
            midpoint = int(.5 * (row.start + row.end))

            seq = self.genome[row.chrom][ midpoint - self.context_length//2:midpoint + self.context_length//2]
            yield(one_hot(seq), np.float32(1)) # positive example

            if prev_chrom == row.chrom and prev_end < row.start:
                midpoint = int(.5 * (prev_end + row.start))
                seq = self.genome[row.chrom][ midpoint - self.context_length//2:midpoint + self.context_length//2]
                yield(one_hot(seq), np.float32(0)) # negative example midway inbetween peaks, could randomize

            prev_chrom = row.chrom
            prev_end = row.end


def dataloader(datadir, dataname, seq_len, batch_size, num_workers):

    genome = pickle.load(open(datadir+"/hg19.pickle","rb"))

    if dataname == 'atac':

        atac_data = pd.read_csv(
            datadir + "/ATAC_data.bed.gz", sep='\t', names=("chrom", "start", "end")
        )

        atac_data = atac_data.sort_values(['chrom', 'start'])  # actually already sorted but why not

        validation_chromosomes = ["chr3", "chr4"]
        validation_data = atac_data[
            atac_data['chrom'].isin(validation_chromosomes)
        ]

        train_data = atac_data[
            ~atac_data['chrom'].isin(validation_chromosomes)
        ]

    elif dataname == 'chip':

        binding_data = pd.read_csv(
            datadir + "/ENCFF300IYQ.bed.gz",
            sep='\t',
            usecols=range(6),
            names=("chrom","start","end","name","score","strand")
            )

        binding_data = binding_data[~binding_data['chrom'].isin(["chrX","chrY"])]
        binding_data = binding_data.sort_values(
            ['chrom', 'start']).drop_duplicates()

        validation_chromosomes = ["chr2","chr3"]
        validation_data = binding_data[
            binding_data['chrom'].isin(validation_chromosomes)
            ]

        train_chromosomes = ["chr%i" % i for i in range(4, 22+1)]
        train_data = binding_data[
            binding_data['chrom'].isin(train_chromosomes)
            ]

    train_dataset = BedPeaksDataset(
        train_data,
        genome,
        seq_len
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
        )

    validation_dataset = BedPeaksDataset(
        validation_data,
        genome,
        seq_len
        )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size
        )

    return train_dataloader, validation_dataloader
