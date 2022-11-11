#!/usr/bin/env python
# coding: utf-8
from os import chdir
import pandas as pd
import sys


def set_control_file_root():
    control_file_dir = '/home/markgreenneuroscience_gmail_com/ML/RANDOM_FOREST/RANDOM_FOREST_CF'
    return control_file_dir


def run_all_generate_random_forest_cf() -> None:
    control_file_dir = set_control_file_root()
    random_forest_control = pd.DataFrame(list(range(25, 525, 25)))
    random_forest_control.columns = ['ITERATORS']
    chdir(control_file_dir)
    random_forest_control.to_csv('random_forest_cf.txt', index=False, header=False)
    sys.exit('RANDOM FOREST CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_generate_random_forest_cf()
