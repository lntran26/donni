import dadi
import numpy as np
import pickle
import os
import sys
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # generate parameter list for training (nu full range)
    # exclude params where T/nu > 5 version
    train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
                          for T in np.linspace(0.1, 2, 20) if T/nu <= 5]

#     # do not exclude param version
#     train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
#                           for T in np.linspace(0.1, 2, 20)]

    # print training set info 
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))
    
    # generate a list of theta values to run scaling and add variance
    theta_list = [1,100,1000,10000]

    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # Use function to make lists of dictionaries storing different training 
    # data sets from lists of parameters
    list_train_dict = util.generating_data_parallel(train_params, 
                        theta_list, func, ns, pts_l)

    # Save training set as a pickle file
    pickle.dump(list_train_dict, open('data/1d-2epoch/train-data-exclude', 'wb'), 2)
#     pickle.dump(list_train_dict, open('data/1d-2epoch/train-data-full', 'wb'), 2)


