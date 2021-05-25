
"""
model params are in log-scale base 10
between models, must change 
    train_params 
    test_params
    func
    ns
    logs
    pnames
"""

# import sys
import time
import dadi
import numpy as np
import random
import util
import matplotlib.pyplot as plt
import pickle
import math
from scipy import stats

if __name__ == '__main__': 
    list_train_dict = pickle.load(open('bin/neural_network/train_set','rb'))
    list_train_dict = [list_train_dict[2], list_train_dict[0]] # 1000 and 1
    list_test_dict = pickle.load(open('bin/neural_network/test_set', 'rb'))
    list_test_dict = [list_test_dict[0], list_test_dict[2]] # 1 and 1000

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

    fig3=plt.figure(3, figsize=(14,13.5), dpi=300)
    plt.title("T", fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.weight': 'bold'})

    fig4=plt.figure(4, figsize=(15.5,13.5), dpi=300)
    plt.title("m", fontsize=50 , fontweight='bold')
    plt.axis("off")
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.weight': 'bold'})


    count_pos = 1

    for train_dict in list_train_dict:
        nn = util.nn_train(train_dict)

        for test_dict in list_test_dict:
            y_true, y_predict = util.nn_test(nn, test_dict)
            param_true, param_pred = util.sort_by_param(y_true, y_predict)
            r2_by_param = util.nn_r2_score(y_true, y_predict)[1]

            plt.figure(1)
            fig1.add_subplot(2, 2, count_pos)
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
            util.plot_by_param(param_true[2], param_pred[2], r2_by_param[2])
            
            plt.figure(4)
            fig4.add_subplot(2, 2, count_pos)
            util.plot_by_param(param_true[3], param_pred[3], r2_by_param[3])
            count_pos += 1

    fig1.savefig('results/2d-splitmig/poster/nu1_nn.png', bbox_inches='tight')
    fig2.savefig('results/2d-splitmig/poster/nu2_nn.png', bbox_inches='tight')
    fig3.savefig('results/2d-splitmig/poster/T_nn.png', bbox_inches='tight')
    fig4.savefig('results/2d-splitmig/poster/m_nn.png', bbox_inches='tight')