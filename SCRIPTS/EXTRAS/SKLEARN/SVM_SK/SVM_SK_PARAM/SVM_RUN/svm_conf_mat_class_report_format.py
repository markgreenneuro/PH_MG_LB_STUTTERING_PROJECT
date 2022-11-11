#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd


def add_labels(dataframe, splits, splitsstring):
    [r, _] = dataframe.shape
    feat_num_column = pd.DataFrame(np.repeat(splits, r))
    feat_num_column.columns = [splitsstring]
    dataframe = pd.concat([feat_num_column, dataframe], axis=1)
    return dataframe


def format_class_report_conf_mat(cint, knl, gma, dfs, conf_mat, class_report):
    classes = np.array(pd.DataFrame(class_report).columns)[:-3]
    classes_df = pd.DataFrame(classes, columns=['CLASSES'])
    class_report = pd.DataFrame(class_report).reset_index()
    class_report = class_report.rename(columns={class_report.columns[0]: 'KEY'})
    class_report.columns = class_report.columns.str.replace(' ', '_')

    class_report = add_labels(class_report, cint, 'CINT')
    class_report = add_labels(class_report, knl, 'KNL')
    class_report = add_labels(class_report, gma, 'GAMMA')
    class_report = add_labels(class_report, dfs, 'DFS')

    conf_mat = pd.DataFrame(conf_mat, columns=classes)
    conf_mat = pd.concat([classes_df, conf_mat], axis=1)
    conf_mat = add_labels(conf_mat, cint, 'CINT')
    conf_mat = add_labels(conf_mat, knl, 'KNL')
    conf_mat = add_labels(conf_mat, gma, 'GAMMA')
    conf_mat = add_labels(conf_mat, dfs, 'DFS')
    conf_mat.columns = conf_mat.columns.str.upper()

    return [conf_mat, class_report]
