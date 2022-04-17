'''Module for using trained MLPR to make demographic param predictions'''
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
import numpy as np
import dadi
import sys
import os

def predict(models: list, input_fs, logs, mapie=True):
    '''
    models: list of single mlpr object if sklearn,
        list of multiple mlpr objects if mapie
    input_fs: single Spectrum object from which to generate prediction

    if mapie, should be passing in a list of models trained on
        individual params
    if not mapie, should be list of length 1
    '''
    input_x = [np.array(input_fs).flatten()]
    if mapie:
        pred_list = []
        for model in models:
            pred_list.append(model.predict(input_x)[0])
    else: # don't know if this works yet
        pred_list = models[0].predict([input_x])
    pred_list = [10**pred_list[i] if logs[i] else pred_list[i] for i in range(len(logs))]
    return pred_list