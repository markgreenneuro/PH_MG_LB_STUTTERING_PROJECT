#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from joblib import dump
from os import chdir, makedirs
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import label_binarize


from random_forest_conf_mat_class_report_format import *
from random_forest_conf_mat_class_report_output import *
from random_forest_summary_stats import *


def get_params():
    #splits = int(sys.argv[1])
    splits: int = 30
    return splits


def set_data_dir():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_files(data_dir):
    chdir(data_dir)
    x_y_train = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_train.csv')
    x_y_test = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_test.csv')
    return [x_y_train, x_y_test]


def split_x_y_train_test(x_y_train, x_y_test):
    x_train = x_y_train.loc[:, x_y_train.columns != 'stutter']
    x_test = x_y_test.loc[:, x_y_test.columns != 'stutter']
    y_train = x_y_train[['stutter']]
    y_test = x_y_test[['stutter']]
    return [x_train, x_test, y_train, y_test]


def randomforest_classifier(x_train, x_test, y_train, y_test, random_state, splits):
    clf = RandomForestClassifier(n_estimators=splits, n_jobs=-1, random_state=random_state)
    clf.fit(x_train, np.array(y_train).ravel())
    y_pred = clf.predict(x_test)
    y_test = np.array(y_test).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    return [y_test, y_pred, clf]


def classification_statistics(y_test, y_pred):
    class_report = classification_report(y_test, y_pred, labels=np.unique(y_pred), output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))
    return [class_report, conf_mat]


def get_num_classes(x_y_train):
    stutter = x_y_train[['stutter']]
    num_classes = len(np.unique(stutter))
    return num_classes


def compute_roc_and_auc(y_test, y_pred, x_y_train):
    fpr = dict()
    tpr = dict()
    y_test = label_binarize(y_test, classes=np.unique(x_y_train.stutter))
    y_pred = label_binarize(y_pred, classes=np.unique(x_y_train.stutter))
    roc_auc = dict()
    for i in range(0, len(np.unique(x_y_train.stutter))):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    roc_auc = roc_auc['micro']
    roc_auc = pd.DataFrame([roc_auc], columns=["ROC_AUC"])
    return roc_auc


def get_num_params(clf):
    num_params = clf.n_features_ + 1
    return num_params


def calculate_aic(n, mse, num_params):
    if mse == 0:
        aic = 2 * num_params
    else:
        aic = n * np.log(mse) + 2 * num_params
    return aic


def calculate_bic(n, mse, num_params):
    if mse == 0:
        bic = num_params * np.log(n)
    else:
        bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def get_mse_bic_aic(y_test, y_pred, clf):
    num_params = get_num_params(clf)
    mse = mean_absolute_error(y_test, y_pred)
    bic = calculate_bic(len(y_test), mse, num_params)
    aic = calculate_aic(len(y_test), mse, num_params)
    mse = pd.DataFrame([mse], columns=['MSE'])
    bic = pd.DataFrame([bic], columns=['BIC'])
    aic = pd.DataFrame([aic], columns=['AIC'])
    return [mse, bic, aic]


def get_statistics(y_test, y_pred, num_classes, clf):
    [class_report, conf_mat] = classification_statistics(y_test, y_pred)
    roc_auc = compute_roc_and_auc(y_test, y_pred, num_classes)
    [mse, bic, aic] = get_mse_bic_aic(y_test, y_pred, clf)
    return [class_report, conf_mat, roc_auc, mse, bic, aic]


def set_results_dir():
    results_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS')
    if results_dir.exists():
        pass
    else:
        makedirs(results_dir)
    results_dir = str(results_dir)
    return results_dir


def set_stats_dir(results_dir):
    stats_dir = Path.home().joinpath(results_dir, 'RANDOM_FOREST_STATS')
    if stats_dir.exists():
        pass
    else:
        makedirs(stats_dir)
    stats_dir = str(stats_dir)
    return stats_dir


def set_model_output(results_dir):
    model_dir = Path.home().joinpath(results_dir, 'RANDOM_FOREST_MODELS')
    if model_dir.exists():
        pass
    else:
        makedirs(model_dir)
    model_dir = str(model_dir)
    return model_dir


def save_model(model_dir, clf, splits):
    chdir(model_dir)
    model_name = str(splits) + '.model'
    dump(clf, model_name)


def run_all_random_forest() -> None:
    start_time = datetime.now()
    random_state=1234
    splits = get_params()
    data_dir = set_data_dir()
    [x_y_train, x_y_test] = get_files(data_dir)
    [x_train, x_test, y_train, y_test] = split_x_y_train_test(x_y_train, x_y_test)
    [y_test, y_pred, clf] = randomforest_classifier(x_train, x_test, y_train, y_test, random_state, splits)
    [class_report, conf_mat, roc_auc, mse, bic, aic] = \
        get_statistics(y_test, y_pred, x_y_train, clf)
    results_dir = set_results_dir()
    stats_dir = set_stats_dir(results_dir)
    [conf_mat, class_report] = format_class_report_conf_mat(conf_mat, class_report, splits)
    output_conf_mat_class_report(splits, conf_mat, class_report, stats_dir)
    output_summary_stats(roc_auc, mse, bic, aic, splits, stats_dir)
    model_dir = set_model_output(results_dir)
    save_model(model_dir, clf, splits)
    time_delta = datetime.now() - start_time
    exit_message = 'RANDOM FOREST COMPLETED SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_all_random_forest()
