#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, LSTM, Conv1D, InputLayer, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.python.ops.init_ops import glorot_uniform_initializer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras_tuner as kt
from create_output_dir import *
from instantiate_data import *
from initialise_settings_and_dim_x import *
from standard_scaler import *
from reformat_data import *
from output_models import *
from set_seed import *
#from kesmarag.ml.hmm import HMM
import tensorflow_hmm
from hmm_layer import *

def build_model(hp):
    encoder_decoder = hp.Choice(name='encoder_decoder', values=['conv', 'lstm', 'dense'])

    number_of_layers = hp.Choice(name='number_of_layers', values=[2, 3, 4])

    encoder_ni_neurons = hp.Int(name="encoder_ni_neurons", min_value=32, max_value=128, step=32)
    encoder_nii_neurons = hp.Int(name="encoder_nii_neurons", min_value=32, max_value=128, step=32)
    encoder_niii_neurons = hp.Int(name="encoder_niii_neurons", min_value=32, max_value=128, step=32)
    encoder_niv_neurons = hp.Int(name="encoder_niv_neurons", min_value=32, max_value=128, step=32)

    encoder_ni_neurons_activation = hp.Choice(name="encoder_ni_neurons_activation", values=["elu", "relu", "tanh"])
    encoder_nii_neurons_activation = hp.Choice(name="encoder_nii_neurons_activation", values=["elu", "relu", "tanh"])
    encoder_niii_neurons_activation = hp.Choice(name="encoder_niii_neurons_activation", values=["elu", "relu", "tanh"])
    encoder_niv_neurons_activation = hp.Choice(name="encoder_niv_neurons_activation", values=["elu", "relu", "tanh"])

    encoder_ni_neurons_dropout_bool = hp.Boolean(name="encoder_ni_neurons_dropout_bool")
    encoder_nii_neurons_dropout_bool = hp.Boolean(name="encoder_nii_neurons_dropout_bool")
    encoder_niii_neurons_dropout_bool = hp.Boolean(name="encoder_niii_neurons_dropout_bool")
    encoder_niv_neurons_dropout_bool = hp.Boolean(name="encoder_niv_neurons_dropout_bool")

    encoder_ni_neurons_dropout_value = hp.Float(name="encoder_ni_neurons_dropout_value", min_value=0.0,
                                                max_value=1.0, step=0.1)
    encoder_nii_neurons_dropout_value = hp.Float(name="encoder_nii_neurons_dropout_value", min_value=0.0,
                                                 max_value=1.0, step=0.1)
    encoder_niii_neurons_dropout_value = hp.Float(name="encoder_niii_neurons_dropout_value", min_value=0.0,
                                                  max_value=1.0, step=0.1)
    encoder_niv_neurons_dropout_value = hp.Float(name="encoder_niv_neurons_dropout_value", min_value=0.0,
                                                 max_value=1.0, step=0.1)

    encoder_ni_neurons_batch_normalisation_bool = hp.Boolean(name="encoder_ni_neurons_batch_normalisation_bool")
    encoder_nii_neurons_batch_normalisation_bool = hp.Boolean(name="encoder_nii_neurons_batch_normalisation_bool")
    encoder_niii_neurons_batch_normalisation_bool = hp.Boolean(name="encoder_niii_neurons_batch_normalisation_bool")
    encoder_niv_neurons_batch_normalisation_bool = hp.Boolean(name="encoder_niv_neurons_batch_normalisation_bool")

    encoder_ni_neurons_batch_normalisation_momentum = hp.Float(name="encoder_ni_neurons_batch_normalisation_momentum",
                                                               min_value=0.0, max_value=1.0, step=0.1)
    encoder_ni_neurons_batch_normalisation_epsilon = hp.Float(name="encoder_ni_neurons_batch_normalisation_epsilon",
                                                              min_value=0.0, max_value=1.0, step=0.1)
    encoder_nii_neurons_batch_normalisation_momentum = hp.Float(name="encoder_nii_neurons_batch_normalisation_momentum",
                                                                min_value=0.0, max_value=1.0, step=0.1)
    encoder_nii_neurons_batch_normalisation_epsilon = hp.Float(name="encoder_nii_neurons_batch_normalisation_epsilon",
                                                               min_value=0.0, max_value=1.0, step=0.1)
    encoder_niii_neurons_batch_normalisation_momentum = hp.Float(
        name="encoder_niii_neurons_batch_normalisation_momentum",
        min_value=0.0, max_value=1.0, step=0.1)
    encoder_niii_neurons_batch_normalisation_epsilon = hp.Float(name="encoder_niii_neurons_batch_normalisation_epsilon",
                                                                min_value=0.0, max_value=1.0, step=0.1)
    encoder_niv_neurons_batch_normalisation_momentum = hp.Float(name="encoder_niv_neurons_batch_normalisation_momentum",
                                                                min_value=0.0, max_value=1.0, step=0.1)
    encoder_niv_neurons_batch_normalisation_epsilon = hp.Float(name="encoder_niv_neurons_batch_normalisation_epsilon",
                                                               min_value=0.0, max_value=1.0, step=0.1)

    decoder_ni_neurons_activation = hp.Choice(name="decoder_ni_neurons_activation", values=["relu", "elu", "tanh"])
    decoder_ni_neurons_dropout_bool = hp.Boolean(name="decoder_ni_neurons_dropout_bool")
    decoder_ni_neurons_dropout_value = hp.Float(name="decoder_ni_neurons_dropout_value", min_value=0.0, max_value=1.0,
                                                step=0.01)
    decoder_ni_neurons_batch_normalisation_bool = hp.Boolean(name="decoder_ni_neurons_batch_normalisation_bool")
    decoder_ni_neurons_batch_normalisation_momentum = hp.Float(name="decoder_ni_neurons_batch_normalisation_momentum",
                                                               min_value=0.0, max_value=1.0, step=0.1)
    decoder_ni_neurons_batch_normalisation_epsilon = hp.Float(name="decoder_ni_neurons_batch_normalisation_epsilon",
                                                              min_value=0.0, max_value=1.0, step=0.1)

    lr_value = hp.Choice(name="lr_value", values=[1e-3, 1e-4, 1e-5])

    optimizer_clipnorm_float_value = hp.Float(name="optimizer_clipnorm_float_value", min_value=0.0, max_value=1.0,
                                              step=0.1)
    optimizer_momentum_float_value = hp.Float(name="optimizer_momentum_float_value", min_value=0.0, max_value=1.0,
                                              step=0.1)
    loss = hp.Choice(name="loss", values=["hmm", "softmax"])

    model = call_existing_code(encoder_decoder, number_of_layers,
                               encoder_ni_neurons, encoder_nii_neurons, encoder_niii_neurons,
                               encoder_niv_neurons,
                               encoder_ni_neurons_activation, encoder_nii_neurons_activation,
                               encoder_niii_neurons_activation,
                               encoder_niv_neurons_activation, encoder_ni_neurons_dropout_bool,
                               encoder_nii_neurons_dropout_bool,
                               encoder_niii_neurons_dropout_bool, encoder_niv_neurons_dropout_bool,
                               encoder_ni_neurons_dropout_value, encoder_nii_neurons_dropout_value,
                               encoder_niii_neurons_dropout_value, encoder_niv_neurons_dropout_value,
                               encoder_ni_neurons_batch_normalisation_bool,
                               encoder_nii_neurons_batch_normalisation_bool,
                               encoder_niii_neurons_batch_normalisation_bool,
                               encoder_niv_neurons_batch_normalisation_bool,
                               encoder_ni_neurons_batch_normalisation_momentum,
                               encoder_ni_neurons_batch_normalisation_epsilon,
                               encoder_nii_neurons_batch_normalisation_momentum,
                               encoder_nii_neurons_batch_normalisation_epsilon,
                               encoder_niii_neurons_batch_normalisation_momentum,
                               encoder_niii_neurons_batch_normalisation_epsilon,
                               encoder_niv_neurons_batch_normalisation_momentum,
                               encoder_niv_neurons_batch_normalisation_epsilon,
                               decoder_ni_neurons_activation, decoder_ni_neurons_dropout_bool,
                               decoder_ni_neurons_dropout_value, decoder_ni_neurons_batch_normalisation_bool,
                               decoder_ni_neurons_batch_normalisation_momentum,
                               decoder_ni_neurons_batch_normalisation_epsilon,
                               lr_value, optimizer_clipnorm_float_value,
                               optimizer_momentum_float_value, loss)

    return model


def call_existing_code(encoder_decoder, number_of_layers,
                       encoder_ni_neurons, encoder_nii_neurons, encoder_niii_neurons,
                       encoder_niv_neurons,
                       encoder_ni_neurons_activation, encoder_nii_neurons_activation,
                       encoder_niii_neurons_activation,
                       encoder_niv_neurons_activation, encoder_ni_neurons_dropout_bool,
                       encoder_nii_neurons_dropout_bool,
                       encoder_niii_neurons_dropout_bool, encoder_niv_neurons_dropout_bool,
                       encoder_ni_neurons_dropout_value, encoder_nii_neurons_dropout_value,
                       encoder_niii_neurons_dropout_value, encoder_niv_neurons_dropout_value,
                       encoder_ni_neurons_batch_normalisation_bool,
                       encoder_nii_neurons_batch_normalisation_bool,
                       encoder_niii_neurons_batch_normalisation_bool,
                       encoder_niv_neurons_batch_normalisation_bool,
                       encoder_ni_neurons_batch_normalisation_momentum,
                       encoder_ni_neurons_batch_normalisation_epsilon,
                       encoder_nii_neurons_batch_normalisation_momentum,
                       encoder_nii_neurons_batch_normalisation_epsilon,
                       encoder_niii_neurons_batch_normalisation_momentum,
                       encoder_niii_neurons_batch_normalisation_epsilon,
                       encoder_niv_neurons_batch_normalisation_momentum,
                       encoder_niv_neurons_batch_normalisation_epsilon,
                       decoder_ni_neurons_activation, decoder_ni_neurons_dropout_bool,
                       decoder_ni_neurons_dropout_value, decoder_ni_neurons_batch_normalisation_bool,
                       decoder_ni_neurons_batch_normalisation_momentum,
                       decoder_ni_neurons_batch_normalisation_epsilon,
                       lr_value, optimizer_clipnorm_float_value,
                       optimizer_momentum_float_value,
                       loss):
    # create model
    data_dir = '/scratch/users/k1754828/DATA/'
    data = InstantiateData(data_dir)
    init_sets = InitialiseSettings(seed_value=1234)
    data = DimXNumCats(data, init_sets)
    data = ReformatData(data, batch_size=batch_size)
    SetSeed(data)

    #tf.keras.backend.clear_session()

    if encoder_decoder == 'conv':
        try:
            del model
        except NameError:
            pass
        tf.keras.backend.clear_session()
        SetSeed(data)
        model = Sequential()
        model.add(InputLayer(input_shape=(14000, data.dim_x), name="input_conv"))
        model.add(Conv1D(encoder_ni_neurons, data.num_cats, activation=encoder_ni_neurons_activation, padding='same', strides=1, kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_ni_neurons_layer_conv"))
        if encoder_ni_neurons_dropout_bool == "encoder_ni_neurons_dropout_bool":
            model.add(Dropout(encoder_ni_neurons_dropout_value))
        if encoder_ni_neurons_batch_normalisation_bool == "encoder_ni_neurons_batch_normalisation":
            model.add(BatchNormalization(momentum=encoder_ni_neurons_batch_normalisation_momentum,
                                         epsilon=encoder_ni_neurons_batch_normalisation_epsilon))
        model.add(Conv1D(encoder_nii_neurons, data.num_cats, activation=encoder_nii_neurons_activation, padding='same', strides=1,kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_nii_neurons_layer_conv"))
        if encoder_nii_neurons_dropout_bool == "encoder_nii_neurons_dropout_bool":
            model.add(Dropout(encoder_nii_neurons_dropout_value))
        if encoder_nii_neurons_batch_normalisation_bool == "encoder_nii_neurons_batch_normalisation":
            model.add(BatchNormalization(momentum=encoder_nii_neurons_batch_normalisation_momentum,
                                         epsilon=encoder_nii_neurons_batch_normalisation_epsilon))
        model.add(
            Conv1D(encoder_niii_neurons, data.num_cats, activation=encoder_niii_neurons_activation, padding='same', strides=1, kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_niii_neurons_layer_conv"))
        if number_of_layers >= 3:
            if encoder_niii_neurons_dropout_bool == "encoder_niii_neurons_dropout_bool":
                model.add(Dropout(encoder_niii_neurons_dropout_value))
            if encoder_niii_neurons_batch_normalisation_bool == "encoder_niii_neurons_batch_normalisation":
                model.add(BatchNormalization(momentum=encoder_niii_neurons_batch_normalisation_momentum,
                                             epsilon=encoder_niii_neurons_batch_normalisation_epsilon))
        else:
            pass
        if number_of_layers == 4:

            model.add(
                Conv1D(encoder_niv_neurons, data.num_cats, activation=encoder_niv_neurons_activation, padding='same', strides=1, kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_niv_neurons_layer_conv"))
            if encoder_niv_neurons_dropout_bool == "encoder_niv_neurons_dropout_bool":
                model.add(Dropout(encoder_niv_neurons_dropout_value))
            if encoder_niv_neurons_batch_normalisation_bool == "encoder_niv_neurons_batch_normalisation_bool":
                model.add(BatchNormalization(momentum=encoder_niv_neurons_batch_normalisation_momentum,
                                             epsilon=encoder_niv_neurons_batch_normalisation_epsilon))
        else:
            pass
        model.add(Conv1D(data.num_cats, 1, activation='elu', padding='same', strides=1, name='decoder_ni_neurons_layer_conv'))
        model.add(HMMLayer(states=data.num_cats, length=data.length, name='hmmlayer_conv'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(learning_rate=lr_value, clipnorm=optimizer_clipnorm_float_value,
                                               momentum=optimizer_momentum_float_value))

    elif encoder_decoder == 'lstm':
        try:
            del model
        except NameError:
            pass
        tf.keras.backend.clear_session()
        SetSeed(data)
        model = Sequential()
        model.add(InputLayer(input_shape=(14000, data.dim_x), name="input_lstm"))
        model.add(LSTM(encoder_ni_neurons, return_sequences=True,kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_ni_neurons_layer_lstm"))
        if encoder_ni_neurons_dropout_bool == "encoder_ni_neurons_dropout_bool":
            model.add(Dropout(encoder_ni_neurons_dropout_value))
        if encoder_ni_neurons_batch_normalisation_bool == "encoder_ni_neurons_batch_normalisation_bool":
            model.add(BatchNormalization(momentum=encoder_ni_neurons_batch_normalisation_momentum,
                                         epsilon=encoder_ni_neurons_batch_normalisation_epsilon))
        model.add(LSTM(encoder_nii_neurons, return_sequences=True,kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_nii_neurons_layer_lstm"))
        if encoder_nii_neurons_dropout_bool == "encoder_nii_neurons_dropout_bool":
            model.add(Dropout(encoder_nii_neurons_dropout_value))
        if encoder_nii_neurons_batch_normalisation_bool == "encoder_nii_neurons_batch_normalisation_bool":
            model.add(BatchNormalization(momentum=encoder_nii_neurons_batch_normalisation_momentum,
                                         epsilon=encoder_nii_neurons_batch_normalisation_epsilon))
        if number_of_layers >= 3:
            model.add(LSTM(encoder_niii_neurons, return_sequences=True,kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_niii_neurons_layer_lstm"))
            if encoder_niii_neurons_dropout_bool == "encoder_niii_neurons_dropout_bool":
                model.add(Dropout(encoder_niii_neurons_dropout_value))
            if encoder_niii_neurons_batch_normalisation_bool == "encoder_niii_neurons_batch_normalisation_bool":
                model.add(BatchNormalization(momentum=encoder_niii_neurons_batch_normalisation_momentum,
                                             epsilon=encoder_niii_neurons_batch_normalisation_epsilon))
        else:
            pass
        if number_of_layers == 4:
            model.add(LSTM(encoder_niv_neurons, return_sequences=True,kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros', name="encoder_niv_neurons_layer_lstm"))
            if encoder_niv_neurons_dropout_bool == "encoder_niv_neurons_dropout_bool":
                model.add(Dropout(encoder_niv_neurons_dropout_value))
            if encoder_niv_neurons_batch_normalisation_bool == "encoder_niv_neurons_batch_normalisation_bool":
                model.add(BatchNormalization(momentum=encoder_niv_neurons_batch_normalisation_momentum,
                                             epsilon=encoder_niv_neurons_batch_normalisation_epsilon))
        else:
            pass
        model.add(LSTM(data.num_cats, return_sequences=True,kernel_initializer=glorot_uniform_initializer(),
                    bias_initializer='zeros'))
        model.add(HMMLayer(states=data.num_cats, length=data.length, name='hmmlayer_lstm'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(learning_rate=lr_value, clipnorm=optimizer_clipnorm_float_value,
                                               momentum=optimizer_momentum_float_value))

    elif encoder_decoder == 'dense':
        try:
            del model
        except NameError:
            pass
        tf.keras.backend.clear_session()
        SetSeed(data)
        model = Sequential()
        model.add(InputLayer(input_shape=(14000, data.dim_x), name="input_dense"))
        model.add(Dense(encoder_ni_neurons,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        name='encoder_ni_neurons_layer_dense'))
        model.add(Activation(encoder_ni_neurons_activation))
        if encoder_ni_neurons_dropout_bool == "ecoder_ni_neurons_dropout_bool":
            model.add(Dropout(encoder_ni_neurons_dropout_value))
        if encoder_ni_neurons_batch_normalisation_bool == "ecoder_ni_neurons_batch_normalisation_bool":
            model.add(BatchNormalization(momentum=encoder_ni_neurons_batch_normalisation_momentum,
                                         epsilon=encoder_ni_neurons_batch_normalisation_epsilon))
        model.add(Dense(encoder_nii_neurons,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        name='encoder_ni_neurons_layer_dense'))
        model.add(Activation(encoder_nii_neurons_activation))
        if encoder_nii_neurons_dropout_bool == "encoder_nii_neurons_dropout_bool":
            model.add(Dropout(encoder_nii_neurons_dropout_value))
        if encoder_nii_neurons_batch_normalisation_bool == "encoder_nii_neurons_batch_normalisation_bool":
            model.add(BatchNormalization(momentum=encoder_nii_neurons_batch_normalisation_momentum,
                                         epsilon=encoder_nii_neurons_batch_normalisation_epsilon))
        if number_of_layers >= 3:
            model.add(Dense(encoder_niii_neurons,
                            kernel_initializer=glorot_uniform_initializer(),
                            bias_initializer='zeros',
                            name='encoder_niii_neurons_layer_dense'))
            model.add(Activation(encoder_niii_neurons_activation))
            if encoder_niii_neurons_dropout_bool == "encoder_niii_neurons_dropout_bool":
                model.add(Dropout(encoder_niii_neurons_dropout_value))
            if encoder_niii_neurons_batch_normalisation_bool == "encoder_niii_neurons_batch_normalisation_bool":
                model.add(BatchNormalization(momentum=encoder_niii_neurons_batch_normalisation_momentum,
                                             epsilon=encoder_niii_neurons_batch_normalisation_epsilon))
        else:
            pass

        if number_of_layers == 4:
            model.add(Dense(encoder_niv_neurons,
                            kernel_initializer=glorot_uniform_initializer(),
                            bias_initializer='zeros',
                            name='encoder_niv_neurons_layer_dense'))
            model.add(Activation(encoder_niv_neurons_activation))
            if encoder_niv_neurons_dropout_bool == "encoder_niv_neurons_dropout_bool":
                model.add(Dropout(encoder_niv_neurons_dropout_value))
            if encoder_niv_neurons_batch_normalisation_bool == "encoder_niv_neurons_batch_normalisation_bool":
                model.add(BatchNormalization(momentum=encoder_niv_neurons_batch_normalisation_momentum,
                                             epsilon=encoder_niv_neurons_batch_normalisation_epsilon))
        model.add(Dense(data.num_cats,
                        kernel_initializer=glorot_uniform_initializer(),
                        bias_initializer='zeros',
                        name='decoder_ni_neurons_layer'))
        model.add(Activation(decoder_ni_neurons_activation))
        if decoder_ni_neurons_dropout_bool == "decoder_ni_neurons_dropout_bool":
            model.add(Dropout(decoder_ni_neurons_dropout_value))
        if decoder_ni_neurons_batch_normalisation_bool == "decoder_ni_neurons_batch_normalisation_bool":
            model.add(BatchNormalization(momentum=decoder_ni_neurons_batch_normalisation_momentum,
                                         epsilon=decoder_ni_neurons_batch_normalisation_epsilon))
        model.add(HMMLayer(states=data.num_cats, length=data.length, name='hmmlayer_dense'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(learning_rate=lr_value, clipnorm=optimizer_clipnorm_float_value,
                                               momentum=optimizer_momentum_float_value))

    sys.stdout.write(str(model.summary()))

    return model


class run_tuner_get_best_hyperparameters:
    def __init__(self, instantiate_data, max_epochs, markov_chain_dir, min_delta, *args, **kwargs):
        super(run_tuner_get_best_hyperparameters, self).__init__(*args, **kwargs)
        self.max_epochs = max_epochs
        self.markov_chain_models = markov_chain_dir.markov_chain_models
        self.markov_chain_tensorboard = markov_chain_dir.markov_chain_tensorboard
        self.markov_chain_phase_models = markov_chain_dir.markov_chain_models
        self.min_delta = min_delta
        self.run_tuner(instantiate_data)

    def run_tuner(self, instantiate_data):
        self.tuner = kt.Hyperband(build_model,
                                  objective='val_accuracy',
                                  max_epochs=self.max_epochs,
                                  factor=3,
                                  # distribution_strategy=tf.distribute.MirroredStrategy(),
                                  overwrite=False,
                                  directory=self.markov_chain_models,
                                  project_name='markov_chain_pretrain')
        tensorflow_board = tf.keras.callbacks.TensorBoard(self.markov_chain_tensorboard)
        partial_models = tf.keras.callbacks.ModelCheckpoint(filepath=self.markov_chain_models +
                                                                     '/model.{epoch:02d}.h5')
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=self.min_delta, patience=5)
        self.tuner.search(instantiate_data.xytrain, validation_data=instantiate_data.xytest,
                          callbacks=[stop_early, partial_models, tensorflow_board])
        return self

#RUN_ALL
data_dir = '/scratch/users/k1754828/DATA/'
results_dir = '/users/k1754828/RESULTS/'
markov_chain_dir = CreateOutputDirectory(results_dir)

min_delta = 0.0001
batch_size = 14000
max_epochs = 10000

data = InstantiateData(data_dir)
init_sets = InitialiseSettings(seed_value=1234)
data = DimXNumCats(data, init_sets)
data = ConductSklearnStandardScaling(data)
data = ReformatData(data, batch_size=batch_size)


run = run_tuner_get_best_hyperparameters(data, max_epochs, markov_chain_dir, min_delta)
OutputModel(markov_chain_dir, run)
