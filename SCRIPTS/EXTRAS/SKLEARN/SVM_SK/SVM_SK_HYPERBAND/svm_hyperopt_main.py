from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials, space_eval
import mlflow
import pandas as pd
import os
from pathlib import Path


class instantiate_data:
    def __init__(self, *args, **kwargs):
        super(instantiate_data, self).__init__(*args, **kwargs)
        self.data_dir = '/home/'
        self.set_data_root(self.data_dir)
        xytrain, xytest = self.get_files()
        self.split_files(xytrain, xytest)

    def set_data_root(self, data_dir):
        self.data_dir = Path(data_dir)
        return self

    def get_files(self):
        os.chdir(self.data_dir)
        xytrain = pd.read_csv('train.csv')
        xytest = pd.read_csv('test.csv')
        return xytrain, xytest

    def split_files(self, xytrain, xytest):
        self.ytrain = xytrain['stutter']
        self.xtrain = xytrain.loc[:, xytrain.columns != 'stutter']
        self.ytest = xytest['stutter']
        self.xtest = xytest.loc[:, xytest.columns != 'stutter']
        return self


class get_dim_x:
    def __init__(self, instantiate_data, *args, **kwargs):
        super(get_dim_x, self).__init__(*args, **kwargs)
        self.dim_x = pd.DataFrame(instantiate_data.data.xtrain).shape[1]


class scale_data:
    def __init__(self, instantiate_data, *args, **kwargs):
        super(scale_data, self).__init__(*args, **kwargs)
        self.conduct_sklearn_standard_scaling(instantiate_data)


class tuner_search_over_parameters:
    def __init__(self, instantiate_data, *args, **kwargs):
        super(tuner_search_over_parameters, self).__init__(*args, **kwargs)
        self.conduct_sklearn_standard_scaling(instantiate_data)


def objective(params):
    data = instantiate_data()
    data = scale_data(data)
    data = get_dim_x(data)
    clf = SVC(**params)
    clf.fit(data.x_train, data.y_train)
    accuracy = cross_val_score(clf, data.xtest, data.ytest).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}


def run_search_over_svm_parameters():
    search_space = [
        {
            'C': hp.uniform('C', 0.0, 1.0),
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid', 'precomputedf']),
            'degree': hp.choice('degree', [1, 2, 3]),
            'gamma': hp.choice('gamma', ['scale', 'auto',
                                         ','.join(list(np.array(np.array(list(range(0, 11, 1)))
                                                                / 10).astype(str)))]),
            'coef0': hp.uniform('coef0', 0.0, 1.0),
            'shrinking': hp.choice('shrinking', [True, False]),
            'verbose': hp.choice('verbose', [True, False]),
            'max_iter': hp.choice('max_iter', [1000000]),
            'decision_function_shape': hp.choice('decision_function_shape', ['ovo', 'ovr'])
        },
    ]

    algo = tpe.suggest

    spark_trials = SparkTrials()

    with mlflow.start_run():
        best_result = fmin(
            fn=objective,
            space=search_space,
            algo=algo,
            max_evals=32,
            trials=spark_trials)

    best_result_df = space_eval(search_space, best_result)
    return best_result, best_result_df


class output_model:
    def __init__(self, results_dir, best_result, *args, **kwargs):
        super(output_model, self).__init__(*args, **kwargs)
        self.set_model_output(results_dir)
        self.save_model(best_result)

    def set_model_output(self, results_dir):
        self.model_dir = Path.home().joinpath(results_dir, 'svm_winning_features_model')
        if self.model_dir.exists():
            pass
        else:
            os.makedirs(self.model_dir)
        self.model_dir = str(self.model_dir)
        return self.model_dir

    def save_model(self, best_result):
        os.chdir(self.model_dir)
        self.model = best_result
        self.model.save('svm_tuner_best.model')


class output_selected_features:
    def __init__(self, results_dir, best_result_df, *args, **kwargs):
        super(output_selected_features, self).__init__(*args, **kwargs)
        self.set_winning_feats_dir(results_dir)
        self.best_result_df = best_result_df
        self.save_winning_features()

    def set_winning_feats_dir(self, results_dir):
        self.winning_feats_dir = Path.home().joinpath(results_dir, 'svm_winning_features_dataframe')
        if self.winning_feats_dir.exists():
            pass
        else:
            self.os.makedirs(self.winning_feats_dir)
        self.winning_feats_dir = str(self.winning_feats_dir)
        return self.winning_feats_dir

    def save_winning_features(self):
        os.chdir(self.winning_feats_dir)
        self.best_result_df.to_csv('svm_tuner_best_model.csv')


def run_svm_tuner_full_script():
    results_dir = '/home'
    best_result, best_result_df = run_search_over_svm_parameters()
    output_model(results_dir, best_result)
    output_selected_features(results_dir, best_result_df)


# RUN_ALL
run_svm_tuner_full_script()

