import os
import sys
import time
import dadi
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # load testing data set
    list_test_dict = pickle.load(open('data/2d-splitmig/test-data','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-fixed-m','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-vary-T','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-vary-T-100','rb'))
    # load list of trained rfr
    list_rfr = pickle.load(open('data/2d-splitmig/list-rfr','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-sampling','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-fixed-m','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-10','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-10-sampling','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-finer-Tm','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-100','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-100-sampling','rb'))

    test_dict = list_test_dict[0]
    rfr = list_rfr[0]
    y_true, y_pred = util.rfr_test(rfr, test_dict)
    # # print(type(y_true))
    # # print(type(y_pred))
    # # print(len(y_true))
    # # print(len(y_pred)) # list class of len 100 (tested with 100 value)
    # for true, pred in zip(y_true, y_pred):
    #     print('Expected params: ', str(true), 
    #         ' vs. Predict params: ', str(pred))
    # check score
    print('R2 score for each predicted param:', 
                r2_score(y_true, y_pred, multioutput='raw_values'))
    print('Aggr. R2 score for all predicted params:', 
                r2_score(y_true, y_pred),'\n')
    # # load training data set to inspect
    # list_train_dict = pickle.load(open('data/2d-splitmig/train-data','rb'))