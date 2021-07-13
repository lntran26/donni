"""
model params are in log-scale base 10
"""

# import sys
import time
import dadi
import numpy as np
import random
import util
import matplotlib.pyplot as plt
import pickle


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
    
    maxTnu = 5 # for exclusion
    # generate parameter list for training
    train_params = []
    while len(train_params) < 400:
        nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        if T/10**nu <= maxTnu:
            train_params.append((round(nu, 2), round(T, 2)))
    # print training set info 
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))
    
    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test sets
    while (len(test_params) < 100):
    # generate random nu and T within the same range as training data range
        nu = (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        if T/10**nu <= maxTnu: 
            params = (round(nu, 2), round(T, 2))
            test_params.append(params)
    # print testing set info 
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))

    # generate a list of theta values to run scaling and add variance
    theta_list = [1, 100, 1000, 10000]
    print('Theta list:', theta_list)

    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]
    logs = [True, False] # only nu is in log-scale
    pnames = ["nu", "T"]

    list_train_dict = util.generating_data_parallel_log(train_params, 
                        theta_list, func, ns, pts_l, logs)
    list_test_dict = util.generating_data_parallel_log(test_params, 
                        theta_list, func, ns, pts_l, logs)
    
    size = len(theta_list)
    
    fig1, axs1 = plt.subplots(size, size, figsize=(4*size, 4*size))
    fig2, axs2 = plt.subplots(size, size, figsize=(4*size, 4*size))

    train_i = 0
    for train_dict in list_train_dict:
        print("Training with theta = ", theta_list[train_i])
        nn = util.nn_train(train_dict, solver='lbfgs', layers=(200,))
        pickle.dump(nn, open(f'mlpr_{theta_list[train_i]}', 'wb'), 2)
        test_i = 0
        for test_dict in list_test_dict:
            y_true, y_predict = util.nn_test(nn, test_dict)
            param_true, param_pred = util.sort_by_param(y_true, y_predict)
            r2_by_param = util.nn_r2_score(y_true, y_predict)[1]
            # create a plot for each test case

             # for coloring
            T_true = [i[1] for i in y_true]
            nu_true = [i[0] for i in y_true]
            vals = [T/10**nu for T, nu in zip(T_true, nu_true)]  
            
            log_nu_true = param_true[0]
            log_nu_pred =  param_pred[0]
            util.plot_by_param_log(log_nu_true, log_nu_pred, True, axs1[train_i, test_i],
                            r2=r2_by_param[0], case=["nu", theta_list[test_i]], vals=vals)
            
            util.plot_by_param_log(param_true[1], param_pred[1], False, axs2[train_i, test_i],
                         r2=r2_by_param[1], case=["T", theta_list[test_i]], vals=vals)
            
        
            
            test_i += 1
        train_i += 1
    #fig, ax = plt.subplots()
    #ax.plot(nn.loss_curve_)

    fig1.tight_layout(rect=[0,0,1,0.95])
    fig2.tight_layout(rect=[0,0,1,0.95])    
    #fig1.savefig(f'nn_nu_{timestr}.png')
    #fig2.savefig(f'nn_T_{timestr}.png')
    plt.show()
    print("END")
