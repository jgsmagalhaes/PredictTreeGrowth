import argparse
import csv
import gc
import json
import matplotlib
import numpy
import os
import sys
matplotlib.use("TkAgg")
from keras.backend import tensorflow_backend as K
from keras.layers import Activation, Dense, LSTM, ReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from math import sqrt
from matplotlib import pyplot
from numpy.random import seed
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow import set_random_seed

""" Parse Args """
parser = argparse.ArgumentParser()
parser.add_argument("--hyperparameters", help="Parameters", default='hyperparameters.json')
args = parser.parse_args()

import builtins
builtins.hyperparameters = args.hyperparameters

from helpers import DataSaver, LossHistory, LSTMModel, preprocess_data
import predict
sys.path.insert(0, '../utils')
import utils

""" Fix seed """
seed(1)
set_random_seed(1)

""" Constants """
# Get parameters from JSON file
PARAM_GRID = None
with open(args.hyperparameters) as f:
    PARAM_GRID = json.load(f, encoding='utf-8')

STARTING_TREE_INDEX = PARAM_GRID['starting_index'] # 1-indexed
TRAIN_TEST_YEAR_SPLIT = PARAM_GRID['train_test_split']
DATASET_PATH = PARAM_GRID['dataset_path']
RESULTS_PATH = PARAM_GRID['results_path']

MODEL_PATH = os.path.join(RESULTS_PATH, 'model.h5')
MODEL_BAK_PATH = os.path.join(RESULTS_PATH, 'model_bak.h5')

FULL_DATASET = utils.read_data(DATASET_PATH, date_column=0, index_column=0)

""" Helper functions """
def get_tree_results_path(tree_index):
    return os.path.join(RESULTS_PATH, 'tree_%d' % (int(tree_index)+1))

""" Main functions """
"""
Main algorithm
"""
def algorithm(lstm_model, tree_index, train_scaled, test_scaled, list_variables, tree_names):
    results_path = get_tree_results_path(tree_index)

    """ Start experiment """
    print('Starting experiment')
    epochs_list = list(range(1, numpy.array(PARAM_GRID['epochs']).max()+1))
    df = DataSaver(epochs_list, columns=['Epoch'])
    df.append_col('Tree name', tree_names[tree_index], index=0)
    """ Format training data """
    X, Y = train_scaled[:, 0:-len(list_variables)], train_scaled[:, -len(list_variables):]
    X = X.reshape(X.shape[0], 1, X.shape[1])

    """ Format validation data """
    val_x, val_y = test_scaled[:, 0:-len(list_variables)], test_scaled[:, -len(list_variables):]
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1])

    """ Create and fit an LSTM model """
    results = model.fit(
        X,
        Y,
        validation_data=(val_x, val_y),
        epochs=PARAM_GRID['epochs'],
        batch_size=PARAM_GRID['batch_size']
    )

    """ Save weights and configs """
    print('Saving weights information')
    config = lstm_model.model.get_config().copy()
    config['input_shape'] = lstm_model.model.input_shape
    config['output_shape'] = lstm_model.model.output_shape
    config['weights'] = list()
    for layer in lstm_model.model.get_weights():
        config['weights'].append(layer.tolist())
    with open(os.path.join(results_path, 'model_configuration.txt'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    """ Generate and save training and validation loss plots for each cross_validation """
    print('Saving results to file')
    losses = results.history

    train_loss = losses['loss']
    test_loss = losses['val_loss']
    metric_rmse = losses['metric_rmse']
    metric_r2_score = losses['metric_r2_score']

    """ Add loss to raw data """
    df.append_col('Train loss estimation', numpy.array(train_loss).flatten())
    df.append_col('Test loss estimation', numpy.array(test_loss).flatten())
    df.append_col('Metric RMSE estimation', numpy.array(metric_rmse).flatten())
    df.append_col('Metric R2 score estimation', numpy.array(metric_r2_score).flatten())

    del losses
    del train_loss
    del test_loss
    del metric_rmse
    gc.collect()

    """ Save raw data """
    result_file_path = os.path.join(RESULTS_PATH, 'training_results.csv')
    df = df.get_dataframe()
    df.to_csv(os.path.join(results_path, 'train_test_loss_epoch_raw.csv'), index=False)
    if not os.path.isfile(result_file_path):
        df.to_csv(result_file_path, header=True, index=False)
    else:
        df.to_csv(result_file_path, mode='a', header=False, index=False)


""" Main function starts here """
if __name__ == "__main__":
    """ Start of script """
    """ Get all the different trees in the dataset """
    tree_names = numpy.unique(FULL_DATASET.values[:, 0])

    """ Create result directories """
    if not os.path.isdir(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    for i in range(len(tree_names)):
        tree_path = get_tree_results_path(i)
        if not os.path.isdir(tree_path):
            os.makedirs(tree_path)

    """ Pick input columns from full dataset """
    print('Picking columns from dataset ' + DATASET_PATH)
    VARIABLES = PARAM_GRID['variables']
    DATASET = FULL_DATASET[VARIABLES]
    VARIABLES = VARIABLES[1:]

    """ Get train, test and forecast values from dataset """
    train_raw_values = DATASET[str(TRAIN_TEST_YEAR_SPLIT[0]):str(TRAIN_TEST_YEAR_SPLIT[1])].values
    test_raw_values = DATASET[str(TRAIN_TEST_YEAR_SPLIT[1]+1):str(TRAIN_TEST_YEAR_SPLIT[2])].values

    """ Preprocess all the data at once """
    print('Preprocessing data')
    encoder = LabelEncoder()
    train_input_data = preprocess_data(train_raw_values.copy(), encoder)
    test_input_data = preprocess_data(test_raw_values.copy(), encoder)

    """ Transform data to be supervised learning """
    print('Transforming data to supervised multivariate model')
    train_supervised = DataFrame()
    test_supervised = DataFrame()
    for (i, tree) in enumerate(tree_names):
        train_tree_data = train_input_data[train_input_data[:, 0] == i, :]
        test_tree_data = test_input_data[test_input_data[:, 0] == i, :]

        train_tree_supervised = utils.timeseries_to_supervised_multivariate(train_tree_data, 1, 1)
        test_tree_supervised = utils.timeseries_to_supervised_multivariate(test_tree_data, 1, 1)

        """ Drop second tree name column and first row with 0 values """
        train_tree_supervised.drop('var1(t)', axis=1, inplace=True)
        test_tree_supervised.drop('var1(t)', axis=1, inplace=True)
        train_tree_supervised.drop(0, inplace=True)
        test_tree_supervised.drop(0, inplace=True)

        train_supervised =  train_supervised.append(train_tree_supervised)
        test_supervised = test_supervised.append(test_tree_supervised)

    train_supervised = train_supervised.values
    test_supervised = test_supervised.values

    """ Create scaler and scale data """
    scaler = utils.create_scaler(train_supervised[:, 1:])

    """ Create model """
    # If old model exists, rename it
    if os.path.isfile(MODEL_PATH):
        if (os.path.isfile(MODEL_BAK_PATH)):
            os.remove(MODEL_BAK_PATH);
        os.rename(MODEL_PATH, MODEL_BAK_PATH)

    history = LossHistory()
    lstm_create = LSTMModel(history)
    model = lstm_create(
        kernel_init=PARAM_GRID['kernel_init'],
        optimizer=PARAM_GRID['optimizer'],
        bias_init=PARAM_GRID['bias_init'],
        batch_size=PARAM_GRID['batch_size'],
        lr=PARAM_GRID['lr'],
        neurons=PARAM_GRID['neurons'],
        input_shape=(TRAIN_TEST_YEAR_SPLIT[1] - TRAIN_TEST_YEAR_SPLIT[0], 1, len(VARIABLES))
    )

    """ Run algorithm on tree data"""
    for (i, tree) in enumerate(tree_names):
        if (i < STARTING_TREE_INDEX-1):
            print('Skipping tree %d' %(i+1))
            continue
        print('Processing tree %d' % (i+1))

        """ Get tree data """
        train_tree_data = train_supervised[train_supervised[:, 0] == i, :]
        test_tree_data = test_supervised[test_supervised[:, 0] == i, :]

        # Remove tree name column
        train_tree_data = train_tree_data[:, 1:]
        test_tree_data = test_tree_data[:, 1:]

        # Scale data
        train_scaled = scaler.transform(train_tree_data)
        test_scaled = scaler.transform(test_tree_data)

        """ Run algorithm """
        algorithm(model, i, train_scaled, test_scaled, VARIABLES, tree_names)

        """ Save model for backup """
        model.save(MODEL_PATH)

        """ Clear variables """
        del train_tree_data
        del test_tree_data
        gc.collect()

    """ Start prediction """
    print('Starting prediction')
    predict.main()

    sys.exit(0)
