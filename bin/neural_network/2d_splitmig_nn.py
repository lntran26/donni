
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
    # open a text file to record experiment results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # sys.stdout = open('results/1d-2epoch-'+ timestr +'.txt', 'w')
    # print header to visually seperate each run
    print('*'*70, '\n')
    # print the date and time of run
    print('EXPERIMENT DATE: ', time.asctime(time.localtime(time.time())))
    # print guide key to intepret the numbers for training and testing cases
    print(
    '''
    Keys for Training/Testing #:
    # 1 : no noise
    # 2 : theta = 100
    # 3 : theta = 1,000 
    # 4 : theta = 10,000
    '''
        )

    theta_list = [1, 100, 1000, 10000]
    #theta_list = [1, 1000]
    print('Theta list:', theta_list)
    '''
    func = dadi.Demographics2D.split_mig
    ns = [20, 20]
    pts_l = [40, 50, 60]
    logs = [True, True, False, False] # nu1 and nu2 are in log-scale
    '''

    #list_train_dict = pickle.load(open('train_set','rb'))
    list_train_dict = pickle.load(open('train_set_exclude','rb'))
    #list_train_dict = [list_train_dict[0], list_train_dict[2]] # 1 and 1000
    #list_test_dict = pickle.load(open('test_set', 'rb'))
    list_test_dict = pickle.load(open('test_set_exclude', 'rb'))
    #list_test_dict = [list_test_dict[0], list_test_dict[2]] # 1 and 1000

    
    size = len(theta_list)
    
    fig1, axs1 = plt.subplots(size, size, figsize=(4*size, 4*size))
    fig2, axs2 = plt.subplots(size, size, figsize=(4*size, 4*size))
    fig3, axs3 = plt.subplots(size, size, figsize=(4*size, 4*size))
    fig4, axs4 = plt.subplots(size, size, figsize=(4*size, 4*size))
    train_i = 0
    #results_list = []
    for train_dict in list_train_dict:
        print("Training with theta = ", theta_list[train_i])
        nn = util.nn_train(train_dict)
        # plot the loss curve after training
        fig, ax = plt.subplots()
        ax.plot(nn.loss_curve_)
        fig.savefig(f'../../results/{timestr}_splitmig_nn_{theta_list[train_i]}_loss.png')
        test_i = 0
    
        for test_dict in list_test_dict:
            y_true, y_predict = util.nn_test(nn, test_dict)
            param_true, param_pred = util.sort_by_param(y_true, y_predict)
            r2_by_param = util.nn_r2_score(y_true, y_predict)[1]
            #rhos = [stats.spearmanr(param_true[i], param_pred[i])[0] for i in range(4)]
            #results_list.append((y_true, y_predict))
            
            #print("Testing with theta = ", theta_list[test_i])
            #print("True params")
            #for param in param_true:
            #    print(param)
            #print("Pred params")
            #for param in param_pred:
            #    print(param)

            util.plot_by_param_log(param_true[0], param_pred[0], True, axs1[train_i, test_i],
                            r2=r2_by_param[0], case=["nu1", theta_list[test_i]])
            
            util.plot_by_param_log(param_true[1], param_pred[1], True, axs2[train_i, test_i], 
                            r2=r2_by_param[1], case=["nu2", theta_list[test_i]])

          
            util.plot_by_param_log(param_true[2], param_pred[2], False, axs3[train_i, test_i],
                            r2=r2_by_param[2], case=["T", theta_list[test_i]])

            
       
            util.plot_by_param_log(param_true[3], param_pred[3], False, axs4[train_i, test_i], 
                            r2=r2_by_param[3], case=["m", theta_list[test_i]])

                
            test_i += 1
            
        train_i += 1
    fig1.tight_layout(rect=[0, 0, 1, 0.95])    
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig4.tight_layout(rect=[0, 0, 1, 0.95]) 

    fig1.savefig(f'../../results/{timestr}_splitmig_nn_nu1.png')
    fig2.savefig(f'../../results/{timestr}_splitmig_nn_nu2.png')
    fig3.savefig(f'../../results/{timestr}_splitmig_nn_T.png')
    fig4.savefig(f'../../results/{timestr}_splitmig_nn_m.png')
    
    #pickle.dump(results_list, open('nn_results', 'wb'), 2)
    print('end')



