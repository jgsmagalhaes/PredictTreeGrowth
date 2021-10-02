import sys
import numpy
import os
import gc
import json
import matplotlib
import csv
import argparse
matplotlib.use("TkAgg")
from keras.layers import Activation, Dense, LSTM, ReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from math import sqrt
from matplotlib import pyplot
from numpy.random import seed
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import set_random_seed
from keras.backend import tensorflow_backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from helpers import metric_rmse, metric_r2_score, preprocess_data
sys.path.insert(0, '../utils')
import utils

import builtins
if hasattr(builtins, "hyperparameters"):
    defaultHyperparamSource = builtins.hyperparameters
else:
    defaultHyperparamSource = 'hyperparameters.json'

""" Parse Args """
parser = argparse.ArgumentParser()
parser.add_argument("--hyperparameters", help="Parameters", default=defaultHyperparamSource)
args = parser.parse_args()

""" Fix seed """
seed(1)
set_random_seed(1)

""" Constants """
# Get parameters from JSON file
PARAM_GRID = None
with open(args.hyperparameters) as f:
    PARAM_GRID = json.load(f, encoding='utf-8')

STARTING_TREE_INDEX = PARAM_GRID['starting_index'] # 1-indexed
DATASET_PATH = PARAM_GRID['dataset_path']
RESULTS_PATH = PARAM_GRID['results_path']

MODEL_PATH = os.path.join(RESULTS_PATH, 'model.h5')

FULL_DATASET = utils.read_data(DATASET_PATH, date_column=0, index_column=0)

""" Helper functions """
def predict(
    lstm_model,
    tree_index,
    forecast_input_data_raw,
    forecast_scaled,
    prediction_years,
    list_variables,
    tree_names,
    scaler
):
    print('Starting prediction')
    forecast_scaled_x = forecast_scaled[:, 0:-len(list_variables)]

    inputs = utils.slice_data_to_batch_size_factor(forecast_scaled_x, PARAM_GRID['batch_size'])
    forecast_input_data = forecast_input_data_raw[:inputs.shape[0]]

    tree_sig_indices = list()
    for (index, variable) in enumerate(list_variables):
        if (
            variable.lower() == 'a' or
            variable.lower() == 's' or
            variable.lower() == 'ele' or
            variable.lower() == 'sp'
        ):
            tree_sig_indices.append(index)

    predictions = list()
    for year in range(prediction_years[0], prediction_years[1]):
        inputs_reshaped = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])
        yhat = lstm_model.predict(inputs_reshaped, batch_size=PARAM_GRID['batch_size'], verbose=1)
        yhat_inverted = utils.invert_scale(scaler, inputs, yhat, len(list_variables))

        """ Add forecasted value to predictions """
        predictions.append(yhat_inverted[0])

        """ Calculate next input """
        inputs = numpy.vstack((inputs[1:], yhat[-1:]))

        """ Copy over values of 'A', 'S', 'Ele' and 'Sp' """
        for index in tree_sig_indices:
            inputs[-1][index] = inputs[0][index]


    """ Clean up memory """
    gc.collect()

    """ Set forecasted value """
    columns = ['Year', 'Tree name'] + list_variables
    predictions = numpy.array(predictions)
    df = DataFrame(
        index=list(range(1, len(predictions)+1)),
        columns=columns
    )
    df['Tree name'] = tree_names[tree_index]
    df['Year'] = list(range(prediction_years[0], prediction_years[1]))
    for index in range(len(list_variables)):
        variable = list_variables[index]
        df[variable] = predictions[:, index]

    """ Add true value and RMSE to data """
    true_val = forecast_input_data[:, 0]
    rmse = numpy.sqrt(
        numpy.mean(
            numpy.square(
                numpy.subtract(
                    predictions[:true_val.shape[0], 0],
                    true_val
                )
            )
        )
    )
    true_val_column = true_val.tolist() + ['-'] * (len(predictions) - true_val.size)
    df.insert(3, 'BAI True Value', true_val_column)
    df.insert(4, 'RMSE', rmse)

    prediction_path = os.path.join(RESULTS_PATH, 'predictions.csv')
    if not os.path.isfile(prediction_path):
        df.to_csv(prediction_path, header=True, index=False)
    else:
        df.to_csv(prediction_path, mode='a', header=False, index=False)

    rmse_df = DataFrame(index=[1], columns=['RMSE'])
    rmse_df['Tree name'] = tree_names[tree_index]
    rmse_df['RMSE'] = rmse

    rmse_path = os.path.join(RESULTS_PATH, 'rmse.csv')
    if not os.path.isfile(rmse_path):
        rmse_df.to_csv(rmse_path, header=True, index=False)
    else:
        rmse_df.to_csv(rmse_path, mode='a', header=False, index=False)

    """ Clean up memory """
    del df
    del predictions
    del true_val
    del rmse
    del true_val_column
    gc.collect()

""" Main function starts here """
def main():
    """ Start of script """
    """ Get the model """
    model = load_model(
    MODEL_PATH,
    custom_objects={'metric_rmse': metric_rmse, 'metric_r2_score': metric_r2_score}
    )
    # try:
    # except:
    #     print('Something went wrong while loading the model or model does not exist at ' + MODEL_PATH)
    #     print('Exiting')
    #     return

    """ Get all the different trees in the dataset """
    tree_names = numpy.unique(FULL_DATASET.values[:, 0])

    """ Create result directories """
    if not os.path.isdir(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    """ Pick input columns from full dataset """
    print('Picking columns from dataset ' + DATASET_PATH)
    VARIABLES = PARAM_GRID['variables']
    DATASET = FULL_DATASET[VARIABLES]
    VARIABLES = VARIABLES[1:]

    """ Get train, test and forecast values from dataset """
    forecast_values = DATASET.values

    """ Preprocess all the data at once """
    print('Preprocessing data')
    encoder = LabelEncoder()
    forecast_input_data = preprocess_data(forecast_values.copy(), encoder)

    """ Transform data to be supervised learning """
    print('Transforming data to supervised multivariate model')
    forecast_supervised = DataFrame()
    for (i, tree) in enumerate(tree_names):
        forecast_tree_data = forecast_input_data[forecast_input_data[:, 0] == i, :]

        forecast_tree_supervised = utils.timeseries_to_supervised_multivariate(forecast_tree_data, 1, 1)

        """ Drop second tree name column and first row with 0 values """
        forecast_tree_supervised.drop('var1(t)', axis=1, inplace=True)
        forecast_tree_supervised.drop(0, inplace=True)

        forecast_supervised =  forecast_supervised.append(forecast_tree_supervised)

    forecast_supervised = forecast_supervised.values

    """ Create scaler and scale data """
    scaler = utils.create_scaler(forecast_supervised[:, 1:])

    """ Start prediction """
    print('Starting prediction')
    for (i, tree) in enumerate(tree_names):
        if (i < STARTING_TREE_INDEX-1):
            print('Skipping tree %d' %(i+1))
            continue
        print('Processing tree %d' % (i+1))

        """ Get tree data """
        forecast_tree_data = forecast_supervised[forecast_supervised[:, 0] == i, :]
        forecast_tree_input_data = forecast_input_data[forecast_input_data[:, 0] == i, :]

        # Remove tree name column
        forecast_tree_data = forecast_tree_data[:, 1:]
        forecast_tree_input_data = forecast_tree_input_data[:, 1:]

        # Scale data
        forecast_scaled = scaler.transform(forecast_tree_data)

        """ Prediction years """
        prediction_years = (1981, 2051) # 2016 to 2050 inclusive

        """ Reset states to prepare for prediction """
        model.model.reset_states()

        """ Predict """
        predict(
            model,
            i,
            forecast_tree_input_data,
            forecast_scaled,
            prediction_years,
            VARIABLES,
            tree_names,
            scaler
        )

        """ Clear variables """
        del forecast_tree_data
        del forecast_tree_input_data
        del forecast_scaled
        gc.collect()

if __name__ == "__main__":
    main()
    sys.exit(0)
