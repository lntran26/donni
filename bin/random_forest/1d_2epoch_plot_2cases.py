import os
import sys
import time
import dadi
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import util

if __name__ == '__main__': 
    # # load list of trained rfr
    # list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-exclude','rb'))

    # load files for with log transform & full range
    # load testing data set
    list_test_dict = pickle.load(open('data/1d-2epoch/test-data-full','rb'))
    list_test_dict_clipped = [list_test_dict[0], list_test_dict[2]]

    # load list of trained rfr
    list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-full-log','rb'))
    list_rfr_clipped = [list_rfr[2], list_rfr[0]]

    # # load files for with log transform & T/nu > 5 exclusion
    # # load testing data set
    # list_test_dict = pickle.load(open('data/1d-2epoch/test-data-exclude-2','rb'))
    # # load list of trained rfr
    # list_rfr = pickle.load(open('data/1d-2epoch/list-rfr-exclude-log','rb'))

    # Create two figures one for nu and one for T
    fig1=plt.figure(1, figsize=(18,14), dpi=300)
    plt.title("ν", fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 28})
    plt.rcParams.update({'font.weight': 'bold'})
    
    fig2=plt.figure(2, figsize=(17.5,14), dpi=300)
    plt.title("T", fontsize=50, fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 28})
    plt.rcParams.update({'font.weight': 'bold'})

    # testing, and plotting
    count_pos = 1
    for rfr in list_rfr_clipped:
        for test_dict in list_test_dict_clipped:
            y_true, y_predict = util.rfr_test(rfr, test_dict)
            # if use log transformed data
            new_y_predict = util.un_log_transform_predict(y_predict, [0])
            # sort results by param                               
            param_true, param_pred = util.sort_by_param(y_true, new_y_predict)
            # make list of T/nu based on param_true values
            T_over_nu = [T/nu for T, nu in zip(param_true[1], param_true[0])]
            # calculate r2 and msle scores
            r2_by_param = util.rfr_r2_score(y_true, new_y_predict)[1]
            
            # PLOT MULTIPLE SUBPLOT VERSION
            plt.figure(1)
            fig1.add_subplot(2, 2, count_pos)
            util.plot_by_param(param_true[0], param_pred[0], 
                            r2_by_param[0], c=T_over_nu)

            plt.figure(2)
            fig2.add_subplot(2, 2, count_pos)
            util.plot_by_param(param_true[1], param_pred[1], 
                            r2_by_param[1], c=T_over_nu)
            count_pos += 1

    fig1.savefig('results/1d-2epoch/poster/nu_full_log_color.png', bbox_inches='tight')
    fig2.savefig('results/1d-2epoch/poster/T_full_log_color.png', bbox_inches='tight')