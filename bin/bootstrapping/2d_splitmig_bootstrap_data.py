import dadi
import numpy as np
import pickle
import random
import util

if __name__ == '__main__': 
    # generate parameter list for testing
    test_params = []

    while (len(test_params) < 200):
    # generate random nu and T within the same range as training data range
        # nu values not in log scale
        # nu1 = 10 ** (random.random() * 4 - 2)
        # nu2 = 10 ** (random.random() * 4 - 2)
        
        nu1 = random.random() * 4 - 2
        nu2 = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        m = random.random() * 9 + 1
        # if (include condition):
        params = (nu1, nu2, T, m)
        test_params.append(params)
    
    # print some info of testing data
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))
    
    # generate a list of theta values to run scaling and add variance
    theta_list = [100, 1000, 10000]

    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    logs = [True, True, False, False]

    # Use function to make lists of dictionaries storing different testing 
    # data sets from lists of parameters
    # list_test_dict = util.generating_data_parallel(test_params, 
    #                     theta_list, func, ns, pts_l)
    list_test_dict = util.generating_data_log_bootstraps(test_params, 
                    theta_list, func, ns, pts_l, logs)

    # Save training set as a pickle file
    pickle.dump(list_test_dict, open('../../data/test_set_bootstraps', 'wb'), 2)
    
    
    
    