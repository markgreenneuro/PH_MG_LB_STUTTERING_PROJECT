#!/usr/bin/env python
# coding: utf-8
from os import chdir
import pandas as pd


def output_conf_mat_class_report(splits, conf_mat, class_report, stats_dir) -> None:
    chdir(stats_dir)
    name_of_conf_mat_save_file: str = 'CONF_MAT_' + str(splits) + '.csv'
    name_of_class_report_save_file: str = 'CLASS_REPORT_' + str(splits) + '.csv'
    conf_mat.columns = conf_mat.columns.str.lower()
    class_report = class_report.columns.str.lower()
    pd.DataFrame(conf_mat).to_csv(name_of_conf_mat_save_file, index=False)
    pd.DataFrame(class_report).to_csv(name_of_class_report_save_file, index=False)
