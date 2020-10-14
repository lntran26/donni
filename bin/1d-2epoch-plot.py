import os
import sys
import time
import dadi
import numpy as np
import random
import matplotlib.pyplot as plt
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # generate parameter list for training
    train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
                          for T in np.linspace(0.1, 2, 20)]
    # print training set info 
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))

    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test set
    for i in range(100):
    # generate random nu and T within the same range as training data range
        nu = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        params = (round(nu, 2), round(T, 1))
        test_params.append(params)
    # print testing set info 
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))

    # generate a list of theta values to run scaling and add variance
    ### TO DO: CAN SEPARATE TRAIN AND TEST THETA LISTS
    theta_list = [1,1000]
    #print('Theta list:', theta_list)
    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # Use function to make lists of dictionaries storing different training and 
    # testing data sets from lists of parameters
    list_train_dict = util.generating_data_parallel(train_params, 
                        theta_list, func, ns, pts_l)
    list_test_dict = util.generating_data_parallel(test_params, 
                        theta_list, func, ns, pts_l)
    
    # training, testing, and plotting
    count_train = 1
    for train_dict in list_train_dict:
        rfr = util.rfr_train(train_dict, -1)
        count_test = 1
        for test_dict in list_test_dict:
            y_true, y_predict = util.rfr_test(rfr, test_dict)
            param_true, param_pred = util.sort_by_param(y_true, y_predict)
            r2_by_param = util.rfr_r2_score(y_true, y_predict)[1]
            # r2_by_param = r2_score(y_true, y_predict, multioutput='raw_values')
            msle_by_param = util.rfr_msle(y_true, y_predict)[1]
            # msle_by_param = mean_squared_log_error(y_true, y_predict, 
            # multioutput='raw_values')
            count_param = 1
            for true, pred, r2, msle in zip(param_true, param_pred, 
            r2_by_param, msle_by_param):
                util.plot_by_param(true, pred, r2, msle)
                plt.savefig('train'+str(count_train)+'test'+str(count_test)+
                'param'+str(count_param)+'.pdf')
                plt.clf()
                count_param+=1
            count_test+=1
        count_train+=1