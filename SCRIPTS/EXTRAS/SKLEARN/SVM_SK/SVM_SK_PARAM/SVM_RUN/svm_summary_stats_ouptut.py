#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import pandas as pd


def output_summary_stats(roc_auc: float, accuracy: float, mse: float, bic: float, aic: float, cint: int, knl: str,
                         gma: str, dfs: str, stats_dir):
    roc_auc_df = pd.DataFrame([roc_auc], columns=['ROC_AUC'])
    accuracy_df = pd.DataFrame([accuracy], columns=['ACCURACY'])
    mse_df = pd.DataFrame([mse], columns=['MSE'])
    bic_df = pd.DataFrame([bic], columns=['BIC'])
    aic_df = pd.DataFrame([aic], columns=['AIC'])
    full_df = pd.concat([roc_auc_df, accuracy_df, mse_df, bic_df, aic_df], axis=1).T.reset_index()
    full_df.columns = ['LABEL', 'VALUE']
    cint_df = pd.DataFrame(np.repeat(np.array([cint]), full_df.shape[0]), columns=['CINT'])
    knl_df = pd.DataFrame(np.repeat(np.array([knl]), full_df.shape[0]), columns=['KNL'])
    gma_df = pd.DataFrame(np.repeat(np.array([gma]), full_df.shape[0]), columns=['GMA'])
    dfs_df = pd.DataFrame(np.repeat(np.array([dfs]), full_df.shape[0]), columns=['DFS'])
    full_df = pd.concat([cint_df, knl_df, gma_df, dfs_df, full_df], axis=1)

    name: str = str(cint) + '_' + str(knl) + '_' + str(gma) + '_' + str(gma) + '_' + str(dfs)
    name_of_save_file: str = 'SUMMARY_STATS_' + str(name) + '.csv'
    os.chdir(stats_dir)
    full_df.to_csv(name_of_save_file, index=False)
