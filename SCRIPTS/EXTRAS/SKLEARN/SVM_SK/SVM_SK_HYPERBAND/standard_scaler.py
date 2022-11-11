#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler


class conduct_sklearn_standard_scaling:
    def __init__(self, instantiate_data, *args, **kwargs):
        super(conduct_sklearn_standard_scaling, self).__init__(*args, **kwargs)#
        self.mf_train=instantiate_data.xtrain
        self.mf_test=instantiate_data.xtest
        self.split()
        self.standardise()

    def __call__(self):
        x_y = self.x_y
        return x_y


    def split(self):
        self.y_train = pd.DataFrame(self.mf_train.stutter)
        self.x_train = pd.DataFrame(self.mf_train.loc[:, self.mf_train.columns != 'stutter'])
        self.y_test = pd.DataFrame(self.mf_test.stutter)
        self.x_test = pd.DataFrame(self.mf_test.loc[:, self.mf_test.columns != 'stutter'])

        return self


    def standardise(self):
        x = pd.DataFrame(self.x_train)
        self.x_columns = x.columns
        if 'master_idx' in x.columns:
            master_idx = x.master_idx
        if 'sess_idx' in x.columns:
            sess_idx = x.sess_idx
        if 'speaker_id' in x.columns:
            speaker_id = x.speaker_id
        if 'sess_id' in x.columns:
            sess_id = x.sess_id
        x = x.loc[:, x.columns != 'master_idx']
        x = x.loc[:, x.columns != 'sess_idx']
        x = x.loc[:, x.columns != 'speaker_id']
        x = x.loc[:, x.columns != 'sess_id']
        scaler = StandardScaler()
        scaler.fit(x)
        x = pd.DataFrame(scaler.transform(x),columns =self.x_columns)
        if 'sess_id' in x.columns:
            x = pd.concat([sess_id, x], axis=1)
        if 'speaker_id' in x.columns:
            x = pd.concat([speaker_id, x], axis=1)
        if 'sess_idx' in x.columns:
            x = pd.concat([sess_idx, x], axis=1)
        if 'master_idx' in x.columns:
            x = pd.concat([master_idx, x], axis=1)
        self.x_train = x
        x = pd.DataFrame(self.x_test)
        if 'master_idx' in x.columns:
            master_idx = x.master_idx
        if 'sess_idx' in x.columns:
            sess_idx = x.sess_idx
        if 'speaker_id' in x.columns:
            speaker_id = x.speaker_id
        if 'sess_id' in x.columns:
            sess_id = x.sess_id
        x = x.loc[:, x.columns != 'master_idx']
        x = x.loc[:, x.columns != 'sess_idx']
        x = x.loc[:, x.columns != 'speaker_id']
        x = x.loc[:, x.columns != 'sess_id']
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.transform(x),columns =self.x_columns)

        if 'sess_id' in x.columns:
            x = pd.concat([sess_id, x], axis=1)
        if 'speaker_id' in x.columns:
            x = pd.concat([speaker_id, x], axis=1)
        if 'sess_idx' in x.columns:
            x = pd.concat([sess_idx, x], axis=1)
        if 'master_idx' in x.columns:
            x = pd.concat([master_idx, x], axis=1)
        self.x_test=x
        return self