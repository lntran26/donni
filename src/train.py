"""Module for training mapie MLPR with dadi-simulated data"""
import pickle
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
import numpy as np


def train(model_spec: dict, model_dir: str, data_dir: str):
    '''
    Input:
        model_spec: dictionary of MLPR architecture specification
        model_dir: output directory to save trained models
        data_dir: path to train data are stored
        where data as a dictionary of training data, where keys are labels
        as tuples and values are dadi fs objects'''

    # Load training data and parse into input and corresponding labels
    data_dict = pickle.load(open(data_dir, 'rb'))

    X_train_input, y_train_label = [], []
    for param, fs in data_dict.items():
        X_train_input.append(np.array(fs).flatten())
        y_train_label.append(param)  # list of tuples

    # parse labels into single list for each param (required for mapie)
    y_train_label_unpack = list(zip(*y_train_label))
    # # for debugging only
    # print(y_train_label_unpack)

    # train a model for each demographic parameter
    mlpr = MLPRegressor()
    mlpr.set_params(**model_spec)
    # # for debugging only
    # print(mlpr.get_params())

    for i, param in enumerate(y_train_label_unpack):
        # param is a label tuple of len(data_dict) for one dem param
        param_predictor = MapieRegressor(mlpr)
        # note: this code will require the same model specification
        # for different demographic params in the same dem model
        # if use data dict containing multi-output label
        param_predictor.fit(X_train_input, param)

        # save separate model for each parameter in the data
        pickle.dump(param_predictor, open(
            f'{model_dir}/param{i+1}_predictor', 'wb'), 2)


if __name__ == "__main__":

    mlpr_param = {'hidden_layer_sizes': (100,),
                  'activation': 'relu', 'solver': 'lbfgs',
                  'alpha': 1, 'max_iter': 5000}
    data = '../tests/test_data/1d_2epoch_100fs_exclude'
    model = '../out'

    train(mlpr_param, model, data)
