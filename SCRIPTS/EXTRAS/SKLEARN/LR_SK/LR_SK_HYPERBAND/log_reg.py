from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
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
        xytrain, xytest=self.get_files()
        self.split_files(xytrain, xytest)


    def set_data_root(self, data_dir):
        self.data_dir = Path(data_dir)
        return self

    def get_files(self):
        os.chdir(self.data_dir)
        xytrain = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_train.csv')
        xytest = pd.read_csv('speech_features_kf_ts_mi_bic_cut_standardised_test.csv')
        return xytrain, xytest

    def split_files(self, xytrain, xytest):
        self.ytrain =xytrain['stutter']
        self.xtrain = xytrain.loc[:, xytrain.columns != 'stutter']
        self.ytest =xytest['stutter']
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
    data=instantiate_data()
    data=scale_data(data)
    data=get_dim_x(data)
    data=scale_data(data)
    clf = LogisticRegression(**params)
    clf.fit(data.x_train, data.y_train)
    accuracy = cross_val_score(clf, data.xtest, data.ytest).mean()
    return {'loss': -accuracy, 'status': STATUS_OK}




def run_search_over_log_reg_parameters():
    search_space = hp.choice('classifier_type', [
        {
            'random_state':hp.choice('random_state',[123]),
            'tol': hp.choice('tol', [1e-3, 1e-4, 1e-5]),
            'dual': hp.choice('dual',[False, True]),
            'fit_intercept': hp.choice('fit_intercept', [False, True]),
            'penalty':hp.choice('penalty',['l1', 'l2', 'elasticnet', 'none']),
            'C': hp.lognormal('LR_C', 0, 1.0),
            'solver': hp.choice('solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']),
            'multi_class': hp.choice('multi_class', ['multinomial'])
        },
    ])


    algo=tpe.suggest

    spark_trials = SparkTrials()

    with mlflow.start_run():
        best_result = fmin(
        fn=objective,
        space=search_space,
        algo=algo,
        max_evals=32,
        trials=spark_trials)

    best_result_df=space_eval(search_space, best_result)
    return best_result, best_result_df



class output_model:
    def __init__(self, results_dir, best_result, *args, **kwargs):
        super(output_model, self).__init__(*args, **kwargs)
        self.set_model_output(results_dir)
        self.save_model(best_result)

    def set_model_output(self, results_dir):
        self.model_dir = Path.home().joinpath(results_dir, 'log_reg_tuner_models')
        if self.model_dir.exists():
            pass
        else:
            self.os.makedirs(self.model_dir)
        self.model_dir = str(self.model_dir)
        return self.model_dir

    def save_model(self, best_result):
        os.chdir(self.model_dir)
        self.model = best_result
        self.model.save('log_reg_tuner_best.model')


class output_selected_features:
    def __init__(self, results_dir, best_result_df, *args, **kwargs):
        super(output_selected_features, self).__init__(*args, **kwargs)
        self.set_winning_feats_dir(results_dir)
        self.best_result_df=best_result_df
        self.save_winning_features()

    def set_winning_feats_dir(self, results_dir):
        self.winning_feats_dir = Path.home().joinpath(results_dir, 'log_reg_tuner_dataframe')
        if self.winning_feats_dir.exists():
            pass
        else:
            self.os.makedirs(self.winning_feats_dir)
        self.winning_feats_dir = str(self.winning_feats_dir)
        return self.winning_feats_dir


    def save_winning_features(self):
        os.chdir(self.winning_feats_dir)
        self.best_result_df.to_csv('log_reg_tuner_best_model.csv')

def run_log_reg_tuner_full_script():
    results_dir='/home'
    best_result, best_result_df=run_search_over_log_reg_parameters()
    output_model(results_dir,best_result)
    output_selected_features(results_dir, best_result_df)

#RUN_ALL
run_log_reg_tuner_full_script()

