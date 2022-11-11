#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd


def output_summary_stats(reg_score, roc_auc, ll, bic, aic, n, penalty_term, c_term, multi_class_term,
                         elastic_net_term, solver_term, stats_dir):
    reg_score_df = pd.DataFrame([reg_score])
    reg_score_df.columns = ['REG_SCORE']
    roc_auc_df = pd.DataFrame([roc_auc])
    roc_auc_df.columns = ['ROC_AUC']
    ll_df = pd.DataFrame([ll])
    ll_df.columns = ['MSE']
    bic_df = pd.DataFrame([bic])
    bic_df.columns = ['BIC']
    aic_df = pd.DataFrame([aic])
    aic_df.columns = ['AIC']
    n_df = pd.DataFrame([n])
    n_df.columns = ['N_VC']

    full_df = pd.concat([roc_auc_df, reg_score_df, ll_df, bic_df, aic_df, n_df], axis=1).T.reset_index()
    full_df.columns = ['LABEL', 'VALUE']
    [r, _] = full_df.shape

    penalty_term_df = pd.DataFrame(np.repeat(penalty_term, r), columns=['PENALTY_TERM'])
    c_term_df = pd.DataFrame(np.repeat(c_term, r), columns=['C_TERM'])
    multi_class_term_df = pd.DataFrame(np.repeat(multi_class_term, r), columns=['MULTI_CLASS_TERM'])
    solver_term_df = pd.DataFrame(np.repeat(solver_term, r), columns=['SOLVER_TERM'])
    labels_df = pd.concat([penalty_term_df, c_term_df, multi_class_term_df, solver_term_df], axis=1)

    if penalty_term == 'elasticnet':
        elastic_net_term_df = pd.DataFrame(np.repeat(elastic_net_term, r), columns=['ELASTIC_NET_TERM'])
        labels_df = pd.concat([labels_df, elastic_net_term_df], axis=1)
    else:
        pass

    name = str(penalty_term) + '_' + str(c_term) + '_' + str(multi_class_term) + '_' + str(solver_term)

    if penalty_term == 'elasticnet':
        name + '_' + str(elastic_net_term)
    else:
        pass

    full_df = pd.concat([labels_df, full_df], axis=1)

    full_df = full_df.loc[:, ~full_df.columns.duplicated()]
    name_of_save_file: str = 'SUMMARY_STATS_' + str(name) + '.csv'
    chdir(stats_dir)
    full_df.to_csv(name_of_save_file, index=False)
