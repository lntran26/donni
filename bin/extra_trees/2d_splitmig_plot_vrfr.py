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
    # load testing data set
    list_test_dict = pickle.load(open('data/2d-splitmig/test-data-corrected','rb'))

    # load list of trained vrfr
    list_vrfr = pickle.load(open('data/2d-splitmig/list-vrfr','rb'))

    # Create four figures for each of the four param
    fig1=plt.figure(1, figsize=(22,16), dpi=300)
    plt.title("nu1-vrfr")
    plt.axis("off")
    
    fig2=plt.figure(2, figsize=(22,16), dpi=300)
    plt.title("nu2-vrfr")
    plt.axis("off")

    fig3=plt.figure(3, figsize=(22,16), dpi=300)
    plt.title("T-vrfr")
    plt.axis("off")

    fig4=plt.figure(4, figsize=(22,16), dpi=300)
    plt.title("m-vrfr")
    plt.axis("off")

    # testing, and plotting
    count_pos = 1
    for vrfr in list_vrfr:
        for test_dict in list_test_dict:
            y_true, y_pred = util.vrfr_test(vrfr, test_dict)
            # sort test results by param for plotting
            param_true, param_pred = util.sort_by_param(y_true, y_pred)
            # calculate r2 and msle scores by param
            r2_by_param = util.vrfr_r2_score(y_true, y_pred)[1]
            # msle_by_param = util.rfr_msle(y_true, y_pred)[1]
            
            # PLOT MULTIPLE SUBPLOT VERSION
            plt.figure(1)
            fig1.add_subplot(4, 4, count_pos)
            # util.plot_by_param(param_true[0], param_pred[0], 
            #                 r2_by_param[0], msle_by_param[0])
            log_nu1_true = [10**param_true for param_true in param_true[0]]
            log_nu1_pred =  [10**param_pred for param_pred in param_pred[0]]
            util.plot_by_param(log_nu1_true, log_nu1_pred, r2_by_param[0])

            plt.figure(2)
            fig2.add_subplot(4, 4, count_pos)
            log_nu2_true = [10**param_true for param_true in param_true[1]]
            log_nu2_pred =  [10**param_pred for param_pred in param_pred[1]]
            util.plot_by_param(log_nu2_true, log_nu2_pred, r2_by_param[1])

            plt.figure(3)
            fig3.add_subplot(4, 4, count_pos)
            # util.plot_by_param(param_true[2], param_pred[2], r2_by_param[2],
            #                 c=[T/m for T,m in zip(param_true[2],param_true[3])])
            util.plot_by_param(param_true[2], param_pred[2], r2_by_param[2])
            
            plt.figure(4)
            fig4.add_subplot(4, 4, count_pos)
            # util.plot_by_param(param_true[3], param_pred[3], r2_by_param[3],
            #                 c=[T/m for T,m in zip(param_true[2],param_true[3])])
            util.plot_by_param(param_true[3], param_pred[3], r2_by_param[3])
            count_pos += 1

    fig1.savefig('results/2d-splitmig/extra_trees/nu1-vrfr.png', bbox_inches='tight')
    fig2.savefig('results/2d-splitmig/extra_trees/nu2-vrfr.png', bbox_inches='tight')
    fig3.savefig('results/2d-splitmig/extra_trees/T-vrfr.png', bbox_inches='tight')
    fig4.savefig('results/2d-splitmig/extra_trees/m-vrfr.png', bbox_inches='tight')