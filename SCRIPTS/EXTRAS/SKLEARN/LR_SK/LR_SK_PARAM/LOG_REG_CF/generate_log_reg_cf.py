#!/usr/bin/env python
# coding: utf-8
from itertools import product
import numpy as np
from os import chdir
import pandas as pd
from pathlib import Path
import sys


def set_control_file_root():
    control_file_dir = Path('/users/k1754828/SCRIPTS/LOG_REG/LOG_REG_CF')
    return control_file_dir


def newton_cg_l2_multinomial():
    solver = 'newton-cg'
    penalty = 'l2'
    multiclass_handling = 'multinomial'
    return [solver, penalty, multiclass_handling]


def liblinear_l2_ovr():
    solver = 'liblinear'
    penalty = 'l2'
    multiclass_handling = 'ovr'
    return [solver, penalty, multiclass_handling]


def cartesian_product(solver, penalty, multiclass_handling):
    solver = np.array([solver])
    penalty = np.array([penalty])
    multiclass_handling = np.array([multiclass_handling])
    c = np.array([x / 10.0 for x in range(0, 11)])
    cartesian_product_df = pd.DataFrame(product(penalty, c, solver, multiclass_handling),
                                        columns=["penalty", "c", "solver", "multiclass"])
    return cartesian_product_df


def stitch(cartesian_product_newton_cg_l2_multinomial, cartesian_product_liblinear_l2_ovr):
    stitched_cartesian_product = pd.concat([cartesian_product_newton_cg_l2_multinomial,
                                            cartesian_product_liblinear_l2_ovr], axis=0).reset_index(drop=True)
    return stitched_cartesian_product


def run_all_generate_logistic_regression_cf(stitched_cartesian_product_name) -> None:
    control_file_dir = set_control_file_root()
    chdir(control_file_dir)
    [solver, penalty, multiclass_handling] = newton_cg_l2_multinomial()
    cartesian_product_newton_cg_l2_multinomial = cartesian_product(solver, penalty, multiclass_handling)
    [solver, penalty, multiclass_handling] = liblinear_l2_ovr()
    cartesian_product_liblinear_l2_ovr = cartesian_product(solver, penalty, multiclass_handling)
    stitched_cartesian_product = stitch(cartesian_product_newton_cg_l2_multinomial, cartesian_product_liblinear_l2_ovr)
    stitched_cartesian_product.to_csv(stitched_cartesian_product_name, index=False, header=False)
    sys.exit('LOG REG CONTROL FILE CREATED SUCCESSFULLY')


# RUN ALL
run_all_generate_logistic_regression_cf('log_reg_cf.txt')
