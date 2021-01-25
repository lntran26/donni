import os
import sys
import time
import dadi
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # # load files for no log transform & full range
    # # load testing data set
    # list_test_dict = pickle.load(open('data/1d-2epoch/test-data-full','rb'))
    # # load list of trained rfr
    # list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-full','rb'))

    # # load files for no log transform & T/nu > 5 exclusion
    # # load testing data set
    # list_test_dict = pickle.load(open('data/1d-2epoch/test-data-exclude','rb'))
    # # load list of trained rfr
    # list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-exclude','rb'))

    # # load files for with log transform & full range
    # # load testing data set
    # list_test_dict = pickle.load(open('data/1d-2epoch/test-data-full','rb'))
    # # load list of trained rfr
    # list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-full-log','rb'))

    # load files for with log transform & T/nu > 5 exclusion
    # load testing data set
    list_test_dict = pickle.load(open('data/1d-2epoch/test-data-exclude-2','rb'))
    # load list of trained rfr
    list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-exclude-log','rb'))


    # generate a list of theta values to run scaling and add variance
    theta_list = [1,100,1000,10000]

    # Create two figures one for nu and one for T
    fig1=plt.figure(1, figsize=(22,16), dpi=300)
    plt.title("nu")
    plt.axis("off")
    
    fig2=plt.figure(2, figsize=(22,16), dpi=300)
    plt.title("T")
    plt.axis("off")

    # testing, and plotting
    count_pos = 1
    for rfr in list_rfr:
        for test_dict in list_test_dict:
            y_true, y_predict = util.rfr_test(rfr, test_dict)
            # if use log transformed data
            new_y_predict = util.un_log_transform_predict(y_predict, [0])
            # if use normal data
            # new_y_predict = y_predict
            # sort results of ML prediction by param                               
            param_true, param_pred = util.sort_by_param(y_true, new_y_predict)
            # make list of T/nu based on param_true values
            T_over_nu = [T/nu for T, nu in zip(param_true[1], param_true[0])]
            # calculate r2 and msle scores
            r2_by_param = util.rfr_r2_score(y_true, new_y_predict)[1]
            msle_by_param = util.rfr_msle(y_true, new_y_predict)[1]
            
            # PLOT MULTIPLE SUBPLOT VERSION
            plt.figure(1)
            fig1.add_subplot(4, 4, count_pos)
            util.plot_by_param(param_true[0], param_pred[0], 
                            r2_by_param[0], msle_by_param[0], T_over_nu)

            plt.figure(2)
            fig2.add_subplot(4, 4, count_pos)
            util.plot_by_param(param_true[1], param_pred[1], 
                            r2_by_param[1], msle_by_param[1], T_over_nu)
            count_pos += 1

    fig1.savefig('results/1d-2epoch/nu_exclude_log_2.png', bbox_inches='tight')
    fig2.savefig('results/1d-2epoch/T_exclude_log_2.png', bbox_inches='tight')