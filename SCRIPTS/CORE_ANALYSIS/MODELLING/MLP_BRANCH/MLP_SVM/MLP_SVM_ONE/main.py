#!/usr/bin/env python
# coding: utf-8
import warnings
import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb
)
from create_svm_output_dir import *
from get_summary_stats import *
from set_seed import *
from output_best_epoch import *
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
from run_best_hps_hyperparameters import *
from output_best_epoch import *

class RunTuneGetBestSvmHyperparametersOne:
    def __init__(self, max_epochs, min_delta, batch_size, seed_value, *args, **kwargs):
        super(RunTuneGetBestSvmHyperparametersOne, self).__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        self.min_delta = min_delta
        self.seed_value = seed_value
        self.batch_size = batch_size

        data = InstantiateData(data_dir='/home/debian/DATA/')
        data = DimXNumCats(data)
        data = ConductSklearnStandardScaling(data)

        data = ReformatData(data, batch_size=14000)
        self.xtrain = data.xtrain
        self.xtest = data.xtest
        self.ytrain = data.ytrain
        self.ytest = data.ytest
        svm_dir = CreateSvmOneDirectory(results_dir='/home/debian/RESULTS')

        self.svm = svm_dir.svm_tf_one_layer
        self.svm_tf_one_layer_partial_models = svm_dir.svm_tf_one_layer_partial_models
        self.svm_tf_one_layer_tensorboard = svm_dir.svm_tf_one_layer_tensorboard
        self.svm_tf_one_layer_pretraining = svm_dir.svm_tf_one_layer_pretraining
        self.run_tuner()

    def run_tuner(self):
        self.tuner = kt.Hyperband(build_model,
                                  objective=kt.Objective('val_accuracy', direction='max'),
                                  max_epochs=self.max_epochs,
                                  factor=3,
                                  # distribution_strategy=tf.distribute.MirroredStrategy(),
                                  overwrite=False,
                                  directory=self.svm_tf_one_layer_pretraining,
                                  project_name='svm_tf_one_layer_tensorboard',
                                  logger=TensorBoardLogger(metrics=["loss", "accuracy", "val_accuracy", "val_loss", ],
                                                           logdir=self.svm_tf_one_layer_pretraining + "/svm_tf_one_layer_tensorboard/hparams")
                                  )
        setup_tb(self.tuner)
        tensorflow_board = tf.keras.callbacks.TensorBoard(self.svm_tf_one_layer_tensorboard)
        partial_models = tf.keras.callbacks.ModelCheckpoint(filepath=self.svm_tf_one_layer_partial_models +
                                                                     '/model.{epoch:02d}.h5')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=self.min_delta,
                                                      patience=5)
        self.tuner.search(self.xtrain, self.ytrain, validation_data=(self.xtest, self.ytest),
                          callbacks=[stop_early, partial_models, tensorflow_board])
        return self

def run_mlp_one():
    start_time = datetime.now()
    batch_size = 14000
    epochs = 1000000
    min_delta = 0.0001

    data = InstantiateData(data_dir='/home/debian/DATA/')
    data = DimXNumCats(data)
    data = ConductSklearnStandardScaling(data)
    data = ReformatData(data, batch_size=14000)
    SetSeed(1234)
    svm_one_dir = CreateSvmOneDirectory(results_dir='/home/debian/RESULTS/')

    RunTuneGetBestSvmHyperparametersOne(max_epochs=epochs, min_delta=min_delta, batch_size=batch_size, seed_value=1234)

    best_hps = run_tuner_get_best_hyperparameters(model_dir=svm_one_dir.svm_tf_one_layer_pretraining,
                                                  project_name='svm_tf_one_layer_tensorboard', epochs=10000)

    epoch_training_model_path = Path(str(svm_one_dir.svm_tf_one_layer_epoch_select_model + '/svm_tf_one_layer_epoch_training_model'))
    if epoch_training_model_path.exists():
        pass
    else:
        model = kt.hypermodel.build(best_hps)
        history = model.fit(data.xytrain, validation_data=(data.xytest), epochs=50)
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
        model.save(str(svm_one_dir.svm_tf_one_layer_epoch_select_model + '/svm_tf_one_layer_epoch_training_model'))
        output_best_epoch.output_best_epoch(svm_one_dir, best_epoch)
        output_best_epoch(svm_one_dir, best_epoch)

    epoch_training_model_path = Path(str(svm_one_dir.svm_tf_one_layer_epoch_select_model + '/svm_tf_one_layer_epoch_training_model'))
    if epoch_training_model_path.exists():
        pass
    else:

        model = kt.hypermodel.build(best_hps)
        # Retrain the model
        model.fit(data.yxtrain, epochs=best_epoch, validation_split=0.2)
        eval_result = model.evaluate(data.xtest, data.xtrain)
        model.save(str(svm_one_dir.svm_tf_one_layer_final_model + '/svm_tf_one_layer_final_model'))
        get_best_epoch(svm_one_dir)

    RunTuneGetBestSvmHyperparametersone()
    best_hps=run_tuner_get_best_hyperparameters(model_dir=svm_one_dir.svm_tf_one_layer_pretraining, project_name='svm_tf_one_layer_tensorboard', epochs=10000)

    epoch_training_model_path = Path(str(svm_one_dir.svm_tf_one_layer_epoch_select_model + '/svm_tf_one_layer_epoch_training_model'))
    if epoch_training_model_path.exists():
        model = tf.keras.models.load_model(str(svm_one_dir.svm_tf_one_layer_epoch_select_model + '/svm_tf_one_layer_epoch_training_model'))
    else:
        model = kt.hypermodel.build(best_hps)
        history = model.fit(data.xtest, validation_data=(data.xtest), epochs=10000, batch_size=14000)
        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))
        model.save(str(svm_one_dir.svm_tf_one_layer_epoch_select_model + '/svm_tf_one_layer_epoch_training_model'))

    epoch_training_model_path = Path(str(svm_one_dir.svm_tf_one_layer_final_model + '/svm_tf_one_layer_final_model'))
    if epoch_training_model_path.exists():
        model = tf.keras.models.load_model(str(svm_one_dir.svm_tf_one_layer_final_model + '/svm_tf_one_layer_final_model'))
    else:
        model = kt.hypermodel.build(best_hps)
        # Retrain the model
        model.fit(data.yxtrain, validation_data=(data.xytest), epochs=best_epoch)
        model.save(str(svm_one_dir.svm_tf_one_layer_final_model + '/svm_tf_one_layer_final_model'))


    data = InstantiateData(data_dir='/home/debian/DATA/')
    data = ConductSklearnStandardScaling(data)
    data = DimXNumCats(data)
    data = ReformatData(data, batch_size)

    test_results = model.evaluate(data.xtest, data.ytest, verbose=1)
    stutter_preds = np.array(model.predict(data.xtest))
    num_classes = len(np.unique(np.argmax(stutter_preds, axis=1)))

    class_report = classification_report(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1))
    class_report = pd.DataFrame([class_report]).transpose()
    conf_mat = confusion_matrix(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1))
    conf_mat = pd.DataFrame(conf_mat)
    test_results = pd.DataFrame([test_results])

    [mse, bic, aic] = get_mse_bic_aic(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1), model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        roc_auc = compute_roc_and_auc(np.argmax(data.ytest, axis=1), np.argmax(stutter_preds, axis=1), num_classes)

    summary_stats = pd.concat([roc_auc, mse, bic, aic], axis=1).T.reset_index()

    best_hps_vals = pd.DataFrame.from_dict(best_hps.values())
    os.chdir(svm_one_dir.svm_tf_one_layer_results)
    test_results.to('test_results.csv', index=False)
    summary_stats.to_csv('summary_stats.csv', index=False)
    pd.DataFrame(conf_mat).to_csv('conf_mat.csv', index=False)
    pd.DataFrame(class_report).to_csv('class_report.csv', index=False)
    best_hps_vals.to_csv('best_hps_vals.csv', index=False)

    time_delta = datetime.now() - start_time

    exit_message = 'SVM one LAYER PARAMETERISED RAN SUCCESSFULLY IN: ' + str(time_delta)
    sys.exit(exit_message)


# RUN ALL
run_mlp_one()
