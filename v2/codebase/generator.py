import torch
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

class Generator():

    def __init__(self, X, y,features, dtypes, task, y_scale,
                 transformation):

        X = X.astype(np.float32)
                
        self.shift = 0

        self.scale = 1

        self.task = task

        self.dtypes = dtypes != 'uint8'

        self.x_tr_va, self.x_te, self.y_tr_va, self.y_te = train_test_split(
            X,
            y,
            test_size=0.1
            )

        kf = KFold()

        self.cv = [
            (
                train_idx,
                valid_idx
                ) for (
                    train_idx,
                    valid_idx
                    ) in kf.split(self.x_tr_va)
                    ]

        self.x_tr = deepcopy(self.x_tr_va)
        self.y_tr = deepcopy(self.y_tr_va)

        self.x_va = deepcopy(self.x_te)
        self.y_va = deepcopy(self.y_te)

        self.k = 0

        self.transformation = transformation

        self.transform()

        self.features = features

        self.y_scale = y_scale

    def set_folds(self, n_splits):

        self.n_splits = n_splits

        if n_splits>1:
            
            # X = np.concatenate([self.x_te, self.x_tr_va])
            # y = np.concatenate([self.y_te, self.y_tr_va])
            
            # self.x_tr_va, self.x_te, self.y_tr_va, self.y_te = train_test_split(
            #     X,
            #     y,
            #     test_size=0.1
            #     )

            kf = KFold(n_splits=n_splits)

            self.cv = [
                (
                    train_idx,
                    valid_idx
                    ) for (
                        train_idx,
                        valid_idx
                        ) in kf.split(self.x_tr_va)
                        ]

            self.k = 0

    def k_fold(self):

        if self.n_splits>1:

            self.inverse_transform()

            train_idx, valid_idx = self.cv[self.k]

            self.x_tr = self.x_tr_va[train_idx, :]
            self.y_tr = self.y_tr_va[train_idx]

            self.x_va = self.x_tr_va[valid_idx, :]
            self.y_va = self.y_tr_va[valid_idx]

            self.k += 1

            self.transform()

    def transform(self):

        if self.transformation == 'minmax':

            self.shift = self.x_tr[:,:-1][:,self.dtypes].min(
                axis=0
                ).reshape(1,-1)

            self.scale = self.x_tr[:,:-1][:,self.dtypes].max(
                axis=0
                ).reshape(1,-1)

            self.scale = (self.x_max - self.x_min)

            self.x_tr[:,:-1][:,self.dtypes] = (self.x_tr[:,:-1][:,self.dtypes]\
                                               - self.shift)/self.scale

            self.x_va[:,:-1][:,self.dtypes] = (self.x_va[:,:-1][:,self.dtypes]\
                                               - self.shift)/self.scale

            self.x_te[:,:-1][:,self.dtypes] = (self.x_te[:,:-1][:,self.dtypes]\
                                               - self.shift)/self.scale

        elif self.transformation == 'normalize':

            self.shift = self.x_tr[:,:-1][:,self.dtypes].mean(
                axis=0
                ).reshape(1,-1)

            self.scale = self.x_tr[:,:-1][:,self.dtypes].std(
                axis=0
                ).reshape(1,-1)

            self.x_tr[:,:-1][:,self.dtypes] = (self.x_tr[:,:-1][:,self.dtypes]\
                                               - self.shift)/self.scale

            self.x_va[:,:-1][:,self.dtypes] = (self.x_va[:,:-1][:,self.dtypes]\
                                               - self.shift)/self.scale

            self.x_te[:,:-1][:,self.dtypes] = (self.x_te[:,:-1][:,self.dtypes]\
                                               - self.shift)/self.scale

        elif self.transformation == 'scale':

            self.scale = self.x_tr[:,:-1][:,self.dtypes].std(
                axis=0
                ).reshape(1,-1)

            self.x_tr[:,:-1][:,self.dtypes] = self.x_tr[:,:-1][:,self.dtypes]\
                /self.scale

            self.x_va[:,:-1][:,self.dtypes] = self.x_va[:,:-1][:,self.dtypes]\
                /self.scale

            self.x_te[:,:-1][:,self.dtypes] = self.x_te[:,:-1][:,self.dtypes]\
                /self.scale

        else:

            pass

    def inverse_transform(self):

        self.x_tr[:,:-1][:,self.dtypes] = self.x_tr[:,:-1][:,self.dtypes]\
            * self.scale + self.shift

        self.x_va[:,:-1][:,self.dtypes] = self.x_va[:,:-1][:,self.dtypes]\
            * self.scale + self.shift

        self.x_te[:,:-1][:,self.dtypes] = self.x_te[:,:-1][:,self.dtypes]\
            * self.scale + self.shift

        self.shift = 0

        self.scale = 1

    def inverse_instance(self, x, covariate=True):

        try:

            x_ = deepcopy(x.detach())

        except:

            x_ = deepcopy(x)

        if covariate:

            try:

                scale = self.scale

                shift = self.shift

                x_[:,:-1][:,self.dtypes] = x_[:,:-1][:,self.dtypes]\
                    * scale + shift

            except:

                scale = torch.tensor(self.scale)

                shift = torch.tensor(self.shift)

                x_[:,:-1][:,self.dtypes] = x_[:,:-1][:,self.dtypes]\
                    * scale + shift

        else:

            try:

                scale = self.scale

                shift = self.shift

                x_[:,self.dtypes] = (x_[:,self.dtypes] - shift) / scale

            except:

                scale = torch.tensor(self.scale)

                shift = torch.tensor(self.shift)

                x_[:,self.dtypes] = (x_[:,self.dtypes] - shift) / scale

        return x_

    def get_x_tr(self, ids):

        return self.x_tr[ids]

    def get_y_tr(self, ids):

        return self.y_tr[ids]

    def get_x_va(self, ids):

        return self.x_va[ids]

    def get_y_va(self, ids):

        return self.y_va[ids]

    def get_x_te(self, ids):

        return self.x_te[ids]

    def get_y_te(self, ids):

        return self.y_te[ids]

    def train_size(self):

        return self.x_tr.shape[0]

    def valid_size(self):

        return self.x_va.shape[0]

    def test_size(self):

        return self.x_te.shape[0]

    def input_size(self):

        return self.x_tr.shape[1]

    def get_task(self):

        return self.task

    def get_y_scale(self):

        return self.y_scale