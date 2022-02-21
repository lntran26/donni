'''Module for using trained MLPR to make demographic param predictions'''
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
import numpy as np


def predict(models: list, test_data: dict, mapie=True, sort=False):
    '''
    models: list of single mlpr object if sklearn,
        list of multiple mlpr objects if mapie

    model: single model
    test_data: fs (arrayLike)

    test_data: actually a list of test dict (several theta case)
    to support both mapie and sklearn similar to train()
    mapie will have several models so test_data have to be the same format
    sort: if True will sort the true and pred values by params-->sklearn only
    no mapie option here because assume the test_data format is compatible
    with each model, but then if use generate_data to make test data
    will have to transform the data outside separately
    if using mapie: should be passing in a list of models

    including out_dir to save output somehow?

    have been using this to get prediction for many tests
    but user likely will only test on their one single case
    then test_data should be just fs, the reference label key should
    be passed in as a separate var
    model should be a single model to predict one thing for one data
    '''
    # unpack test_data dict
    X_test = [np.array(fs).flatten() for fs in test_data.values()]
    y_test = list(test_data.keys())

    # parse labels into single list for each param (required for mapie)
    y_test_unpack = list(zip(*y_test)) if mapie else [y_test]

    # make prediction with trained mlpr models
    for model in models:
        if mapie:
            pred, _ = model.predict(X_test)
        else:
            pred = model.predict([X_test])