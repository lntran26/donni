import dadi
import numpy as np
import pickle
import os
import sys
import random
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test set
    # for i in range(5):
    # for i in range(10):
    for i in range(100):
    # generate random nu and T within the same range as training data range
        # nu values not in log scale
        # nu1 = 10 ** (random.random() * 4 - 2)
        # nu2 = 10 ** (random.random() * 4 - 2)
        
        # nu1 = random.random() * 4 - 2
        nu1 = 1
        # nu2 = random.random() * 4 - 2
        nu2 = 1

        T = random.random() * 1.9 + 0.1
        
        # m = random.random() * 9.9 + 0.1 # this is wrong
        m = random.random() * 9 + 1
        # m = 1
        params = (nu1, nu2, T, m)
        test_params.append(params)
    
    # print some info of testing data
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))
    
    # generate a list of theta values to run scaling and add variance
    theta_list = [1,100,1000,10000]

    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    logs = [True, True, False, False]

    # Use function to make lists of dictionaries storing different testing 
    # data sets from lists of parameters
    # list_test_dict = util.generating_data_parallel(test_params, 
    #                     theta_list, func, ns, pts_l)
    list_test_dict = util.generating_data_parallel_log(test_params, 
                    theta_list, func, ns, pts_l, logs)

    # Save training set as a pickle file
    # pickle.dump(list_test_dict, open('data/2d-splitmig/test-data', 'wb'), 2)
    # pickle.dump(list_test_dict, open('data/2d-splitmig/test-data-fixed-m', 'wb'), 2)
    # pickle.dump(list_test_dict, open('data/2d-splitmig/test-data-vary-T', 'wb'), 2)
    # pickle.dump(list_test_dict, open('data/2d-splitmig/test-data-vary-T-100', 'wb'), 2)
    # pickle.dump(list_test_dict, open('data/2d-splitmig/test-data-corrected', 'wb'), 2)
    pickle.dump(list_test_dict, open('data/2d-splitmig/test-data-vary-T-m', 'wb'), 2)
