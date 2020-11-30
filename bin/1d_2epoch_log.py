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
    for nu in np.linspace(-2, 2, 21):
        for T in np.linspace(0.1, 2, 20):
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


    train_i = 0
    for train_dict in list_train_dict:
        print("Training with theta = ", theta_list[train_i])
        rfr = util.rfr_train(train_dict)
        
        rows, cols = len(theta_list), len(logs)
        fig, axs = plt.subplots(rows, cols, 
                                figsize=(4*cols, 4*rows))
        fig.suptitle(f'Trained on theta={theta_list[train_i]}')
        test_i = 0
        for test_dict in list_test_dict:
            y_true, y_predict = util.rfr_test(rfr, test_dict)
            param_true, param_pred = util.sort_by_param(y_true, y_predict)
            r2_by_param = util.rfr_r2_score(y_true, y_predict)[1]
            # create a plot for each test case
            
            # for coloring
            T_true = [i[1] for i in y_true]
            nu_true = [i[0] for i in y_true]
            vals = [T/10**nu for T, nu in zip(T_true, nu_true)]  
        
            param_i = 0
            for true, pred, log, r2, name in zip(param_true, param_pred,
                                                 logs, r2_by_param, pnames):
                case = (name, theta_list[test_i])
                util.plot_by_param_log(true, pred, log, axs[test_i, param_i],
                                       r2, case, vals)
                param_i += 1
            test_i += 1
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        plt.clf()
        train_i += 1
    print("END")

