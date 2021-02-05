## This script is used to trouble shoot the behavior of RFR on the 2D-splitmig
## case. Goals: inspect each training/testing pair more closely by:
## break each plot into smaller version with coloring to find problem dataset
## inspect the problem data set more closely

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
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data','rb'))
    list_test_dict = pickle.load(open('../../data/2d-splitmig/test-data-corrected','rb'))

    # load list of trained rfr
    list_rfr = pickle.load(open('../../data/2d-splitmig/list-rfr','rb'))

    # # load training data set to inspect
    # list_train_dict = pickle.load(open('data/2d-splitmig/train-data','rb'))

    # choose [0] for theta=1 and [2] for theta=1000
    # test_dict = list_test_dict[0]
    test_dict = list_test_dict[2]
    
    rfr = list_rfr[0]

    y_true, y_pred = util.rfr_test(rfr, test_dict)
    # y_true and y_pred are lists of length 100 (tested with 100 values)
    
    # # print all 100 values of true vs pred
    # for true, pred in zip(y_true, y_pred):
    #     print('Expected params: ', str(true), 
    #         ' vs. Predict params: ', str(pred))

    # # check score
    # print('R2 score for each predicted param:', 
    #             r2_score(y_true, y_pred, multioutput='raw_values'))
    # print('Aggr. R2 score for all predicted params:', 
    #             r2_score(y_true, y_pred),'\n')


    # Break y_true and y_pred into 10 smaller sets each with 10 values
    # How many elements each list should have  
    n = 10
    # Using list comprehension to create a new list with 10 sublists
    y_true_split_10 = [y_true[i:i + n] for i in range(0, len(y_true), n)]  
    y_pred_split_10 = [y_pred[i:i + n] for i in range(0, len(y_pred), n)]  

    # # Plot 10 small sets
    # fig=plt.figure(figsize=(15,35), dpi=300)
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
    #     'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # count_pos = 1
    # for i in range(n):
    #     y_expect = y_true_split_10[i] 
    #     y_predict = y_pred_split_10[i]
    #     param_true, param_pred = util.sort_by_param(y_expect, y_predict)
    #     # nu1=param_[0], nu2=param_[1], T=param_[2], m=param_[3]

    #     fig.add_subplot(10, 4, count_pos)
    #     log_nu1_true = [10**param_true for param_true in param_true[0]]
    #     log_nu1_pred =  [10**param_pred for param_pred in param_pred[0]]
    #     util.plot_by_param(log_nu1_true, log_nu1_pred, c=colors)
    #     count_pos += 1

    #     fig.add_subplot(10, 4, count_pos)
    #     log_nu2_true = [10**param_true for param_true in param_true[1]]
    #     log_nu2_pred =  [10**param_pred for param_pred in param_pred[1]]
    #     util.plot_by_param(log_nu2_true, log_nu2_pred, c=colors)
    #     count_pos += 1

    #     fig.add_subplot(10, 4, count_pos)
    #     util.plot_by_param(param_true[2], param_pred[2], c=colors)
    #     count_pos += 1

    #     fig.add_subplot(10, 4, count_pos)
    #     util.plot_by_param(param_true[3], param_pred[3], c=colors)
    #     count_pos += 1

    # # plt.tight_layout()
    # # plt.show()
    # # fig.savefig('results/2d-splitmig/colors/train1_test1.png', 
    # #                     bbox_inches='tight')
    # # fig.savefig('results/2d-splitmig/colors/train1_test1000.png', 
    # #                     bbox_inches='tight')
    # fig.savefig('results/2d-splitmig/colors/train1_test1_corrected_m.png', 
    #                     bbox_inches='tight')
    # # fig.savefig('results/2d-splitmig/colors/train1_test1000_corrected_m.png', 
    # #                     bbox_inches='tight')

    # print(y_true_split_10[7][9], y_pred_split_10[7][9])
    # print(y_true_split_10[8][1], y_pred_split_10[8][1])
    # print(y_true_split_10[3][1], y_pred_split_10[3][1])
    # print(y_true_split_10[6][0], y_pred_split_10[6][0])
    # print(y_true_split_10[1][9], y_pred_split_10[1][9])
    print(y_true_split_10[5][6], y_pred_split_10[5][6])