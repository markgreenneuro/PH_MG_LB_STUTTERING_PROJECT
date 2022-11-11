#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd


def output_summary_stats(roc_auc: pd.DataFrame, mse: pd.DataFrame, bic: pd.DataFrame, aic: pd.DataFrame, splits: int,
                         stats_dir):
    full_df = pd.concat([roc_auc, mse, bic, aic], axis=1).T.reset_index()
    full_df.columns = ['LABEL', 'VALUE']
    full_df = pd.concat([full_df, pd.DataFrame(np.repeat([splits], full_df.shape[0]), columns=['SPLITS'])], axis=1)
    name_of_save_file: str = 'SUMMARY_STATS_' + str(splits) + '.csv'
    chdir(stats_dir)
    full_df.to_csv(name_of_save_file, index=False)
