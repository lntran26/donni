"""Module for training mapie MLPR with dadi-simulated data"""
import pickle
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
import numpy as np


def train(mlpr_spec: dict, mlpr_dir: str, data_dir: str, mapie=True):
    '''
    Input:
        model_spec: dictionary of MLPR architecture specification
        model_dir: output directory to save trained models
        data_dir: path to train data are stored
        where data as a dictionary of training data, where keys are labels
        as tuples and values are dadi fs objects
        mapie: if False will use sklearn mlpr with multioutput option
        if True (default) will use mapie mlpr with single-output option'''

    # Load training data and parse into input and corresponding labels
    data_dict = pickle.load(open(data_dir, 'rb'))

    # X_train_input, y_train_label = [], []
    # for param, fs in data_dict.items():
    #     X_train_input.append(np.array(fs).flatten())
    #     y_train_label.append(param)  # list of tuples
    X_train_input = [np.array(fs).flatten() for fs in data_dict.values()]
    y_train_label = list(data_dict.keys())
    # test to see if these two versions yield the same results (pyhon3.7+
    # ordered dict so might not be a concern)

    # parse labels into single list for each param (required for mapie)
    y_train_label_unpack = list(
        zip(*y_train_label)) if mapie else [y_train_label]
    # # for debugging only
    # print(y_train_label_unpack)

    # train a model for each demographic parameter
    mlpr = MLPRegressor()
    mlpr.set_params(**mlpr_spec)
    # # for debugging only
    # print(mlpr.get_params())

    for i, param in enumerate(y_train_label_unpack):
        # param is a label tuple of len(data_dict) for one dem param
        param_predictor = MapieRegressor(mlpr) if mapie else mlpr
        # use sklearn mlpr with multi-output or mapie mlpr

        # note: this code will require the same model specification
        # for different demographic params in the same dem model
        # if use data dict containing multi-output label
        param_predictor.fit(X_train_input, param)

        # save separate model for each parameter in the data
        index = i+1 if mapie else 'all'
        pickle.dump(param_predictor, open(
            f'{mlpr_dir}/param_{index}_predictor', 'wb'), 2)


if __name__ == "__main__":

    mlpr_param = {'hidden_layer_sizes': (100,),
                  'activation': 'relu', 'solver': 'lbfgs',
                  'alpha': 1, 'max_iter': 5000}
    data = '../tests/test_data/1d_2epoch_100fs_exclude'
    ml_model = '../out'

    # train(mlpr_param, ml_model, data)
    train(mlpr_param, ml_model, data, mapie=False)
