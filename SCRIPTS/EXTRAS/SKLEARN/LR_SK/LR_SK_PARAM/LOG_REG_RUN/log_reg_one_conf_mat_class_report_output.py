#!/usr/bin/env python
# coding: utf-8
from os import chdir


def output_conf_mat_class_report(penalty_term, c_term, multi_class_term, solver_term, elastic_net_term,
                                 conf_mat, class_report, stats_dir):
    chdir(stats_dir)
    name = str(penalty_term) + '_' + str(c_term) + '_' + str(multi_class_term) + '_' + str(solver_term)
    if penalty_term == 'elasticnet':
        name = name + '_' + str(elastic_net_term)
    else:
        pass
    name_of_conf_mat_save_file: str = 'CONF_MAT_' + str(name) + '.csv'
    name_of_class_report_save_file: str = 'CLASS_REPORT_' + str(name) + '.csv'
    conf_mat.to_csv(name_of_conf_mat_save_file, index=False)
    class_report.to_csv(name_of_class_report_save_file, index=False)
