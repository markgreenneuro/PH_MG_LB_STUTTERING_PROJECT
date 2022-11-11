#!/usr/bin/env python
# coding: utf-8
import numpy as np
from os import chdir
import pandas as pd
import sys


def set_control_file_root():
    control_file_dir = '/users/k1754828/SCRIPTS/SVM/SVM_CF'
    return control_file_dir


def expand_grid(*args):
    mesh = np.meshgrid(*args)
    return pd.DataFrame(m.flatten() for m in mesh)


def control_file_generator():
    c = np.array(list(range(1, 11, 1))) / 10
    kernel = np.array(['rbf'])
    gamma = np.array(['scale', 'auto'])
    decision_function_shape = np.array(['ovr'])
    control_file = expand_grid(c, kernel, gamma, decision_function_shape).T
    control_file.columns = ['C', 'KERNEL', 'GAMMA', 'DECISION_FUNCTION_SHAPE']
    return control_file


def run_all_generate_svm_cf() -> None:
    control_file_dir = set_control_file_root()
    control_file = control_file_generator()
    chdir(control_file_dir)
    control_file.to_csv('svm_cf.txt', sep='\t', header=False, index=False)
    sys.exit('SVM CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_generate_svm_cf()
