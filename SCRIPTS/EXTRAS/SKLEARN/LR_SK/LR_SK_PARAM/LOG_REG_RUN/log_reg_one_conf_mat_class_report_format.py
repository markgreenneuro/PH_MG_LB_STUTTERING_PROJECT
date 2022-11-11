#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


def add_labels(df: pd.DataFrame, label_columns, label: str = 'default'):
    [r, _] = df.shape
    df_labelled = pd.DataFrame(np.repeat(label_columns, r))
    df_labelled.columns = [str(label)]
    df = pd.concat([df_labelled, df], axis=1)
    return df


def class_report_conf_mat(conf_mat, class_report, penalty_term, c_term,
                          multi_class_term, elastic_net_term, solver_term, stutter_test):
    classes = np.array(pd.DataFrame(class_report).columns)[:-3]
    classes_df = pd.DataFrame(classes, columns=['CLASSES'])
    class_report = pd.DataFrame(class_report).reset_index()
    class_report = class_report.rename(columns={class_report.columns[0]: 'KEY'})
    class_report.columns = class_report.columns.str.replace(' ', '_')

    class_report = add_labels(class_report, penalty_term, 'PENALTY_TERM')
    class_report = add_labels(class_report, c_term, 'C_TERM')
    class_report = add_labels(class_report, multi_class_term, 'MULTI_CLASS_TERM')
    class_report = add_labels(class_report, elastic_net_term, 'ELASTIC_NET_TERM')
    class_report = add_labels(class_report, solver_term, 'SOLVER_TERM')
    class_report.columns = class_report.columns.str.upper()

    conf_mat = pd.DataFrame(conf_mat, columns=classes)
    conf_mat = pd.concat([classes_df, conf_mat], axis=1)
    conf_mat = add_labels(conf_mat, penalty_term, 'PENALTY_TERM')
    conf_mat = add_labels(conf_mat, c_term, 'C_TERM')
    conf_mat = add_labels(conf_mat, multi_class_term, 'MULTI_CLASS_TERM')
    conf_mat = add_labels(conf_mat, elastic_net_term, 'ELASTIC_NET_TERM')
    conf_mat = add_labels(conf_mat, solver_term, 'SOLVER_TERM')
    conf_mat.columns = conf_mat.columns.str.upper()

    return [conf_mat, class_report]
