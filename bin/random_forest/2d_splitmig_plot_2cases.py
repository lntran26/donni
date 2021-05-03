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
    list_test_dict = pickle.load(open('data/2d-splitmig/test-data-corrected','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-corrected-2','rb'))
    list_test_dict_clipped = [list_test_dict[0], list_test_dict[2]]

    # load list of trained rfr
    list_rfr = pickle.load(open('data/2d-splitmig/list-rfr','rb'))
    list_rfr_clipped = [list_rfr[2], list_rfr[0]]

    # Create four figures for each of the four param
    fig1=plt.figure(1, figsize=(15,13), dpi=300)
    plt.title(r'$ν_1$', fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.weight': 'bold'})
    
    fig2=plt.figure(2, figsize=(15,13), dpi=300)
    plt.title(r'$ν_2$', fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.weight': 'bold'})

    fig3=plt.figure(3, figsize=(13.5,13), dpi=300)
    plt.title("T", fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.weight': 'bold'})

    fig4=plt.figure(4, figsize=(14,13), dpi=300)
    plt.title("m", fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.weight': 'bold'})

    # testing, and plotting
    count_pos = 1
    for rfr in list_rfr_clipped:
        for test_dict in list_test_dict_clipped:
            y_true, y_pred = util.rfr_test(rfr, test_dict)
            # sort test results by param for plotting
            param_true, param_pred = util.sort_by_param(y_true, y_pred)
            # calculate r2 and msle scores by param
            r2_by_param = util.rfr_r2_score(y_true, y_pred)[1]
            # msle_by_param = util.rfr_msle(y_true, y_pred)[1]
            
            # PLOT MULTIPLE SUBPLOT VERSION
            plt.figure(1)
            fig1.add_subplot(2, 2, count_pos)
            # util.plot_by_param(param_true[0], param_pred[0], 
            #                 r2_by_param[0], msle_by_param[0])
            log_nu1_true = [10**param_true for param_true in param_true[0]]
            log_nu1_pred =  [10**param_pred for param_pred in param_pred[0]]
            util.plot_by_param(log_nu1_true, log_nu1_pred, r2_by_param[0])

            plt.figure(2)
            fig2.add_subplot(2, 2, count_pos)
            log_nu2_true = [10**param_true for param_true in param_true[1]]
            log_nu2_pred =  [10**param_pred for param_pred in param_pred[1]]
            util.plot_by_param(log_nu2_true, log_nu2_pred, r2_by_param[1])

            plt.figure(3)
            fig3.add_subplot(2, 2, count_pos)
            # util.plot_by_param(param_true[2], param_pred[2], r2_by_param[2],
            #                 c=[T/m for T,m in zip(param_true[2],param_true[3])])
            util.plot_by_param(param_true[2], param_pred[2], r2_by_param[2])
            
            plt.figure(4)
            fig4.add_subplot(2, 2, count_pos)
            # util.plot_by_param(param_true[3], param_pred[3], r2_by_param[3],
            #                 c=[T/m for T,m in zip(param_true[2],param_true[3])])
            util.plot_by_param(param_true[3], param_pred[3], r2_by_param[3])
            count_pos += 1

    # # Plot T/m
    # fig1=plt.figure(1, figsize=(22,16), dpi=300)
    # plt.title("T/m")
    # plt.axis("off")

    # # testing, and plotting
    fig1.savefig('results/2d-splitmig/poster/nu1.png', bbox_inches='tight')
    fig2.savefig('results/2d-splitmig/poster/nu2.png', bbox_inches='tight')
    fig3.savefig('results/2d-splitmig/poster/T.png', bbox_inches='tight')
    fig4.savefig('results/2d-splitmig/poster/m.png', bbox_inches='tight')