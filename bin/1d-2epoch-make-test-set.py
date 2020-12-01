import dadi
import numpy as np
import random 
import pickle
import os
import sys
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test set
    for i in range(105):
    # generate random nu and T within the same range as training data range
        nu = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        # # exclude T/nu > 5 version
        # if T/nu <= 5:
        #     params = (nu, T)
        #     test_params.append(params)
        # full range version
        params = (nu, T)
        test_params.append(params)

    # print testing set info 
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))

    # generate a list of theta values to run scaling and add variance
    theta_list = [1,100,1000,10000]

    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # Use function to make lists of dictionaries storing different 
    # testing data sets from lists of parameters
    list_test_dict = util.generating_data_parallel(test_params, 
                        theta_list, func, ns, pts_l)

    # Save testing set as a pickle file
    # pickle.dump(list_test_dict, open('data/1d-2epoch/test-data-exclude', 'wb'), 2)
    pickle.dump(list_test_dict, open('data/1d-2epoch/test-data-full', 'wb'), 2)