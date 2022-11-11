#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from joblib import dump
from os import chdir, makedirs
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, log_loss
from sklearn.preprocessing import label_binarize, LabelEncoder, OneHotEncoder
from tensorflow.keras.losses import CategoricalCrossentropy

from log_reg_one_conf_mat_class_report import *
from log_reg_one_conf_mat_class_output import *
from log_reg_one_summary_stats_output import *


def get_params():
    """
    Belts and braces with explicit cast and object specification.
    :return: [feat_num, penalty_term, dual_term, c_term, multi_class_term, solver_term]
    """
    #penalty_term: str = str(sys.argv[1])
    #c_term: float = float(sys.argv[2])
    #solver_term: str = str(sys.argv[4])
    #multi_class_term: str = str(sys.argv[3])
    #elastic_net_term: float = float(sys.argv[5])

    penalty_term: str = 'l2'
    c_term: float = 0.5
    solver_term: str = 'newton-cg'
    multi_class_term: str = 'multinomial'
    elastic_net_term: str = 'None'



    return [penalty_term, c_term, solver_term, multi_class_term, elastic_net_term]


def set_data_root():
    data_dir = Path('/home/debian/VARIATIONAL_OUTPUT/VARIATIONAL_AUTOENCODER_SPLIT_SETS')
    return data_dir


def get_files(data_dir):
    chdir(data_dir)
    x_y_train = pd.read_csv('speech_features_type_2_283_141_211-10141_141_211_train.csv')
    x_y_test = pd.read_csv('speech_features_type_2_283_141_211-10141_141_211_test.csv')
    return [x_y_train, x_y_test]


def set_log_reg_root():
    log_reg_dir = Path('/home/debian/RESULTS/LOG_REG/')
    if log_reg_dir.exists():
        pass
    else:
        makedirs(log_reg_dir)
    log_reg_dir = str(log_reg_dir)
    return log_reg_dir



def get_make_stats_root(log_reg_dir):
    stats_dir = Path.home().joinpath(log_reg_dir, str('LOG_REG_STATS'))
    if stats_dir.exists():
        pass
    else:
        makedirs(stats_dir)
    stats_dir = str(stats_dir)
    return stats_dir


def logistic_regression(x_y_train, penalty_term, solver_term, c_term, multi_class_term, elastic_net_term):
    stutter_train = x_y_train.stutter
    x_train = x_y_train.loc[:, x_y_train.columns != 'stutter']
    print(multi_class_term)
    print(solver_term)

    if penalty_term == 'elasticnet':
        log_reg = LogisticRegression(penalty=penalty_term, C=c_term,
                                     max_iter=1000000, solver=solver_term, multi_class=multi_class_term,
                                     l1_ratio=elastic_net_term, n_jobs=-1)
    else:
        log_reg = LogisticRegression(penalty=penalty_term, C=c_term,
                                     max_iter=1000000, solver=solver_term, multi_class=multi_class_term, n_jobs=-1)
    log_reg.fit(x_train, np.array(stutter_train))
    return log_reg


def classification_statistics(log_reg, x_y_test):
    stutter_test = x_y_test.stutter
    x_test = x_y_test.loc[:, x_y_test.columns != 'stutter']
    stutter_pred = log_reg.predict(x_test)
    reg_score = log_reg.score(x_test, stutter_test)
    class_report = classification_report(stutter_test, stutter_pred, labels=np.unique(stutter_pred), output_dict=True)
    conf_mat = confusion_matrix(stutter_test, stutter_pred, labels=np.unique(stutter_pred))
    return [reg_score, class_report, stutter_pred, conf_mat]


def get_stutter_test(x_y_test):
    stutter_test = x_y_test.stutter
    return stutter_test


def compute_roc_and_auc(stutter_test, stutter_pred, x_y_train):
    fpr = dict()
    tpr = dict()
    stutter_test = label_binarize(stutter_test, classes=np.unique(x_y_train.stutter))
    stutter_pred = label_binarize(stutter_pred, classes=np.unique(x_y_train.stutter))
    roc_auc = dict()
    for i in range(0, len(np.unique(x_y_train.stutter))):
        fpr[i], tpr[i], _ = roc_curve(stutter_test[:, i], stutter_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # COMPUTE MICRO AVERAGE ROC CURVE
    fpr["micro"], tpr["micro"], _ = roc_curve(stutter_test.ravel(), stutter_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc = roc_auc["micro"]
    return roc_auc


def get_num_params(log_reg):
    num_params = len(log_reg.get_params(deep=True)) + 1
    return num_params


def calculate_aic(n, ll: float, num_params):
    if ll != 0:
        aic = n * np.log(ll) + 2 * num_params
    else:
        aic = 2 * num_params
    return aic


def calculate_bic(n: float, ll: float, num_params: int) -> float:
    if ll == 0:
        bic = num_params * np.log(n)
    else:
        bic = n * np.log(ll) + num_params * np.log(n)
    return bic


def one_hot_encoding(categorical_variable):
    int_encoded = LabelEncoder().fit_transform(categorical_variable)
    int_encoded = int_encoded.reshape(len(int_encoded), -1)
    one_hot_encoded = OneHotEncoder(sparse=False).fit_transform(int_encoded)
    return one_hot_encoded


def get_ll_bic_aic(log_reg,stutter_test, stutter_pred, num_params):
    ll = log_loss(one_hot_encoding(stutter_test), one_hot_encoding(stutter_pred))
    aic = calculate_aic(len(stutter_test), ll, num_params)
    bic = calculate_bic(len(stutter_test), ll, num_params)
    return [ll, aic, bic]


def vapnik_chervonenkis_dimension(x_train):
    d = x_train.shape[1]+1
    return d


def get_y_train_pred(log_reg, x_train):
    y_train_pred = log_reg.predict(x_train)
    return y_train_pred


def get_cross_entropy_error(y1, y2):
    y1_oh = one_hot_encoding(y1)
    y2_oh = one_hot_encoding(y2)
    cross_entropy = CategoricalCrossentropy()
    cross_entropy = np.asarray(cross_entropy(y1_oh, y2_oh))
    cross_entropy_error = 1 - cross_entropy
    return cross_entropy_error


def estimate_number_of_samples(x_y_train, log_reg, delta=0.05):
    x_train = x_y_train.loc[:, x_y_train.columns != 'stutter']
    y_train = x_y_train['stutter']
    d = vapnik_chervonenkis_dimension(x_train)
    y_train_pred = get_y_train_pred(log_reg, x_train)
    epsilon = get_cross_entropy_error(y_train, y_train_pred)
    n = np.ceil((d + np.log(1 / delta)) / epsilon)
    return n


def get_make_modelsave_root(log_reg_dir):
    modelsave_dir = Path.home().joinpath(log_reg_dir, str('LOG_REG_MODELS'))
    if modelsave_dir.exists():
        pass
    else:
        makedirs(modelsave_dir)
    modelsave_dir = str(modelsave_dir)
    return modelsave_dir


def write_modelsave(log_reg_dir, log_reg, penalty_term, c_term, multi_class_term, solver_term, elastic_net_term):
    modelsave_dir = get_make_modelsave_root(log_reg_dir)
    chdir(modelsave_dir)
    name = str(penalty_term) + '_' + str(c_term) + '_' + str(multi_class_term) + '_' + str(solver_term) + '_' + str(elastic_net_term) + '.model'
    dump(log_reg, name)


def run_all_logistic_regression() -> None:
    start_time = datetime.now()
    [penalty_term, c_term, solver_term, multi_class_term, elastic_net_term] = get_params()
    data_dir = set_data_root()
    [x_y_train, x_y_test] = get_files(data_dir)
    log_reg_dir = set_log_reg_root()
    stats_dir = get_make_stats_root(log_reg_dir)
    log_reg = logistic_regression(x_y_train, penalty_term, solver_term, c_term, multi_class_term, elastic_net_term)
    [reg_score, class_report, stutter_pred, conf_mat] = classification_statistics(log_reg, x_y_test)
    stutter_test = get_stutter_test(x_y_test)
    roc_auc = compute_roc_and_auc(stutter_test, stutter_pred, x_y_train)
    num_params = get_num_params(log_reg)
    [ll, aic, bic] = get_ll_bic_aic(log_reg, stutter_test, stutter_pred, num_params)
    n = estimate_number_of_samples(x_y_train, log_reg, delta=0.05)
    [conf_mat, class_report] = class_report_conf_mat(conf_mat, class_report, penalty_term, c_term, multi_class_term,
                                                     elastic_net_term, solver_term, stutter_test)
    output_conf_mat_class_report(penalty_term, c_term, multi_class_term, solver_term, elastic_net_term,
                                 conf_mat, class_report, stats_dir)
    output_summary_stats(reg_score, roc_auc, ll, bic, aic, n, penalty_term, c_term, multi_class_term,
                         elastic_net_term, solver_term, stats_dir)
    write_modelsave(log_reg_dir, log_reg, penalty_term, c_term, multi_class_term, solver_term, elastic_net_term)
    time_delta = datetime.now() - start_time
    exit_message = 'LOGISTIC REGRESSION RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_logistic_regression()
