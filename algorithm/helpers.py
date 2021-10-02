import sys
import keras
import numpy
import keras.optimizers as optimizers
from pandas import DataFrame
from keras.layers import Activation, Dense, LSTM, ReLU
from keras.models import Sequential
from keras.backend import tensorflow_backend as K

sys.path.insert(0, '../utils')
import utils

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.params_list = []
        self.index = 0
        self.model = None

    def on_epoch_end(self, epoch, logs):
        self.model.reset_states()

    def on_model_fit(self, history, model_params):
        self.losses.append(history)
        self.params_list.append(model_params)
        self.index += 1

class DataSaver():
    def __init__(self, data=None, columns=[]):
        self.col_index = 0
        if data is not None:
            self.col_index = 1
        self.df = DataFrame(data, columns=columns)

    def append_col(self, col_name, value, index=None):
        append_index = self.col_index
        if index is not None:
            append_index = index

        size_diff = 0
        is_string = True
        if isinstance(value, str) is False:
            is_string = False
            size_diff = self.df.index.size - value.size
        if (size_diff > 0):
            new_col = value.tolist() + (['-'] * size_diff)
            self.df.insert(append_index, col_name, numpy.array(new_col).flatten())
        elif (size_diff == 0):
            self.df.insert(append_index, col_name, value)
        else:
            self.df.reindex(len(value), fill_value='-')
            self.df.insert(append_index, col_name, value)
        self.col_index += 1

    def get_dataframe(self):
        return self.df

""" LSTM Model Class """
def metric_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def metric_r2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def LSTMModel(history_obj):
    """ Metrics """

    """ Generator """
    def generate_fit_function(old_fit_function):
        def custom_fit_function(
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None
        ):
            """ Slice data such that it's length is a factor of BATCH_SIZE """
            X_sliced = utils.slice_data_to_batch_size_factor(x, batch_size);
            y_sliced = utils.slice_data_to_batch_size_factor(y, batch_size);
            val_data = (
                utils.slice_data_to_batch_size_factor(validation_data[0], batch_size),
                utils.slice_data_to_batch_size_factor(validation_data[1], batch_size),
            )

            history = old_fit_function(
                x=X_sliced,
                y=y_sliced,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks,
                validation_split=validation_split,
                validation_data=val_data,
                shuffle=shuffle,
                class_weight=class_weight,
                sample_weight=sample_weight,
                initial_epoch=initial_epoch,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )
            history_obj.on_model_fit(history.history, history.params)
            return history

        return custom_fit_function

    def generate_evaluate_function(old_evaluate_function):
        def custom_evaluate_function(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
            """ Slice data such that it's length is a factor of BATCH_SIZE """
            X_sliced = utils.slice_data_to_batch_size_factor(x, batch_size)
            y_sliced = utils.slice_data_to_batch_size_factor(y, batch_size)

            return old_evaluate_function(x=X_sliced, y=y_sliced, batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=steps)
        return custom_evaluate_function

    """ Create """
    def create(
        batch_size=1,
        old_model=None,
        neurons=10,
        input_shape=None,
        optimizer='rmsprop',
        lr=0.001,
        kernel_init='random_uniform',
        bias_init='zeros',
    ):
        """ Get optimizer """
        optimizer_obj = None
        if optimizer.lower() == 'adam':
            optimizer_obj = optimizers.Adam(lr=lr)
        elif optimizer.lower() == 'rmsprop':
            optimizer_obj = optimizers.RMSprop(lr=lr)
        elif optimizer.lower() == 'sgd':
            optimizer_obj = optimizers.SGD(lr=lr)
        """ Create model """
        model = Sequential()
        model.add(
            LSTM(
                neurons,
                batch_input_shape=(batch_size, input_shape[1], input_shape[2]),
                stateful=True,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init
            )
        )
        model.add(Dense(input_shape[2]))

        """ Compile model """
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer_obj,
            metrics=[metric_rmse, metric_r2_score]
        )

        """ Override fit and evaluate functions """
        # This is done because we need to split data to have it be a
        # factor of batch size
        model.fit = generate_fit_function(model.fit)
        model.evaluate = generate_evaluate_function(model.evaluate)
        return model

    return create

"""
Preprocess datasets
"""
def preprocess_data(data, label_encoder):
    """ Get input columns from raw data """
    # Remove Tree Name column
    input_data = data

    """ Encode Categorical variables """
    for i,item in enumerate(input_data[0,:]):
        if isinstance(item, str) is True:
            input_data[:, i] = label_encoder.fit_transform(input_data[:, i])
    input_data = input_data.astype('float32')

    """ Return data """
    return input_data
