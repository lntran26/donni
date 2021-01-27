import dadi
import numpy as np
import pickle
import os
import sys
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__':
    # generate parameter list for training
    # train_params = [(nu1, nu2, T, m) for nu1 in 10**np.linspace(-2, 2, 10)
    #                                 for nu2 in 10**np.linspace(-2, 2, 10)
    #                                 for T in np.linspace(0.1, 2, 5)
    #                                 for m in np.linspace(1, 10, 5)]
        
    # # this set keeps m fixed at 1
    # m = 1
    # train_params = [(nu1, nu2, T, m) for nu1 in np.linspace(-2, 2, 10)
    #                             for nu2 in np.linspace(-2, 2, 10)
    #                             for T in np.linspace(0.1, 2, 5)]
    
    # this set fix everything and only vary T
    nu1 = 1
    nu2 = 1
    m = 1
    # train_params = [(nu1, nu2, T, m) for T in np.linspace(0.1, 2, 5)]
    train_params = [(nu1, nu2, T, m) for T in np.linspace(0.1, 2, 100)]

    # # this set has finer grid values for T and m while sacrificing nu1 and nu2
    # train_params = [(nu1, nu2, T, m) for nu1 in np.linspace(-2, 2, 5)
    #                             for nu2 in np.linspace(-2, 2, 5)
    #                             for T in np.linspace(0.1, 2, 10)
    #                             for m in np.linspace(1, 10, 10)]

    # print training set info 
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))

    # generate a list of theta values to run scaling and add variance
    theta_list = [1,100,1000,10000]

    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    logs = [True, True, False, False]

    # Use function to make lists of dictionaries storing different training 
    # data sets from lists of parameters
    # list_train_dict = util.generating_data_parallel(train_params, 
    #                     theta_list, func, ns, pts_l)
    list_train_dict = util.generating_data_parallel_log(train_params, 
                        theta_list, func, ns, pts_l, logs)

    # Save training set as a pickle file
    # pickle.dump(list_train_dict, open('data/2d-splitmig/train-data', 'wb'), 2)
    # pickle.dump(list_train_dict, open('data/2d-splitmig/train-data-fixed-m', 'wb'), 2)
    # pickle.dump(list_train_dict, open('data/2d-splitmig/train-data-vary-T', 'wb'), 2)
    # pickle.dump(list_train_dict, open('data/2d-splitmig/train-data-vary-T-10', 'wb'), 2)
    # pickle.dump(list_train_dict, open('data/2d-splitmig/train-data-finer-Tm', 'wb'), 2)
    pickle.dump(list_train_dict, open('data/2d-splitmig/train-data-vary-T-100', 'wb'), 2)
