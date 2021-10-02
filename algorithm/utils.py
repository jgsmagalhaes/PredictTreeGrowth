from pandas import DataFrame, Series, concat, read_csv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%Y')

def read_data(path, date_column=1, index_column=0):
    return read_csv(path, header=0, parse_dates=[date_column], index_col=index_column, squeeze=True, date_parser=parser)

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

def timeseries_to_supervised_multivariate(data, n_in=1, n_out=1, fillna=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if (fillna):
        agg.fillna(0, inplace=True)
    return agg

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def create_scaler(train):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit(train)

# # scale train and test data to [-1, 1]
# def scale(train, test):
#     # fit scaler
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler = scaler.fit(train)
#     # transform train
#     # train = train.reshape(train.shape[0], train.shape[1])
#     train_scaled = scaler.transform(train)
#     # transform test
#     # test = test.reshape(test.shape[0], test.shape[1])
#     test_scaled = scaler.transform(test)
#     return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value, num_variables=1):
    array = numpy.append(X, value, axis=1)
    inverted = scaler.inverse_transform(array)
    return inverted[:, -num_variables:]

def slice_data_to_batch_size_factor(data, batch_size):
    len_datapoints = int(len(data)/batch_size) * batch_size
    return data[:len_datapoints]

# def convert_json_encoding(input):
#     if isinstance(input, dict):
#         return {convert_json_encoding(key): convert_json_encoding(value) for key, value in input.items()}
#     elif isinstance(input, list):
#         return [convert_json_encoding(element) for element in input]
#     elif isinstance(input, str):
#         return input.encode('utf-8')
#     else:
#         return input
