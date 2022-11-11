#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import sys

from SVM_CONF_MAT_CLASS_REPORT_FORMAT import *
from SVM_CONF_MAT_CLASS_REPORT_OUTPUT import *
from SVM_SUMMARY_STATS_OUTPUT import *


def get_params():
    cint = float(sys.argv[1])
    knl = str(sys.argv[2])
    gma = str(sys.argv[3])
    dfs = str(sys.argv[4])
    return [cint, knl, gma, dfs]


def set_data_dir():
    data_dir = Path('/home/markgreenneuroscience_gmail_com/DATA/MASTER_FEATURE')
    return data_dir


def get_files(data_dir):
    os.chdir(data_dir)
    x_y_train = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_train.csv')
    x_y_test = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_test.csv')
    return [x_y_train, x_y_test]


def split_x_y_train_test(x_y_train, x_y_test):
    x_train = x_y_train.loc[:, x_y_train.columns != 'stutter']
    x_test = x_y_test.loc[:, x_y_test.columns != 'stutter']
    y_train = x_y_train[['stutter']]
    y_test = x_y_test[['stutter']]
    return [x_train, x_test, y_train, y_test]


def classifierfit(x_train, x_test, y_train, y_test, cint, knl, gma, dfs):
    clf = SVC(C=cint, kernel=knl, degree=3, gamma=gma, max_iter=- 1,
              decision_function_shape=dfs, probability=1)
    clf.fit(x_train, np.ravel(y_train))
    accuracy = pd.DataFrame([clf.score(x_test, y_test)])
    accuracy.columns = ['Accuracy']
    y_pred = clf.predict(x_test)
    return [accuracy, y_pred, clf]


def classification_statistics(y_test, y_pred):
    class_report = classification_report(y_test, y_pred, labels=np.unique(y_pred), output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=np.unique(y_pred))
    return [class_report, conf_mat]


def get_num_classes(x_y):
    stutter = x_y[['stutter']]
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
    return roc_auc


def get_num_params(clf):
    num_params = len(clf.get_params(deep=True)) + 1
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
    return [mse, bic, aic]


def get_statistics(x_test, y_test, y_pred, num_classes, clf):
    [class_report, conf_mat] = classification_statistics(x_test, y_test)
    roc_auc = compute_roc_and_auc(y_test, y_pred, num_classes)
    [mse, bic, aic] = get_mse_bic_aic(y_test, y_pred, clf)
    return [class_report, conf_mat, roc_auc, mse, bic, aic]


def run_svm_and_get_statistics(x_train, x_test, y_train, y_test, cint, knl, gma, dfs):
    [accuracy, y_pred, classifier] = classifierfit(x_train, x_test, y_train, y_test, cint, knl, gma, dfs)
    [class_report, conf_mat] = classification_statistics(y_test, y_pred)
    return [y_pred, accuracy, classifier, class_report, conf_mat]


def set_output_dir():
    output_dir = Path('/home/markgreenneuroscience_gmail_com/RESULTS')
    return output_dir


def set_stats_dir(output_dir):
    stats_dir = Path.home().joinpath(output_dir, 'SVM_STATS')
    if stats_dir.exists():
        pass
    else:
        os.makedirs(stats_dir)
    stats_dir = str(stats_dir)
    return stats_dir


def set_model_output(output_dir):
    model_dir = Path.home().joinpath(output_dir, 'SVM_MODELS')
    if model_dir.exists():
        pass
    else:
        os.makedirs(model_dir)
    model_dir = str(model_dir)
    return model_dir


def save_model(model_dir, clf, cint, knl, gma, dfs):
    os.chdir(model_dir)
    model_name = str(cint) + '_' + str(knl) + '_' + str(gma) + '_' + str(dfs) + '.model'
    dump(clf, model_name)


def run_all_svm() -> None:
    start_time = datetime.now()
    [cint, knl, gma, dfs] = get_params()
    data_dir = set_data_dir()
    [x_y_train, x_y_test] = get_files(data_dir)
    [x_train, x_test, y_train, y_test] = split_x_y_train_test(x_y_train, x_y_test)
    [accuracy, y_pred, clf] = classifierfit(x_train, x_test, y_train, y_test, cint, knl, gma, dfs)
    [class_report, conf_mat] = classification_statistics(y_test, y_pred)
    roc_auc = compute_roc_and_auc(y_test, y_pred, x_y_train)
    output_dir = set_output_dir()
    stats_dir = set_stats_dir(output_dir)
    [mse, bic, aic] = get_mse_bic_aic(y_test, y_pred, clf)
    [conf_mat, class_report] = format_class_report_conf_mat(cint, knl, gma, dfs, conf_mat,
                                                            class_report)
    output_conf_mat_class_report(cint, knl, gma, dfs, conf_mat, class_report, stats_dir)
    output_summary_stats(roc_auc, accuracy, mse, bic, aic, cint, knl, gma, dfs, stats_dir)
    model_dir = set_model_output(output_dir)
    save_model(model_dir, clf, cint, knl, gma, dfs)
    time_delta = datetime.now() - start_time
    exit_message = 'SVM COMPLETED SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN_ALL
run_all_svm()
