import math
import numpy as np
import dadi
import random
import pickle
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))  # this is the ml_dadi dir
from data_manip import generating_data


def generate_data_1d_2epoch_unnorm(n_samples, theta_list):
    # generate params
    # save params as dictionary with nu and T as keys
    # and log_theta as values
    params_dict = {}

    while len(params_dict) < n_samples:
        log_theta = random.random() * 3 + 2
        log_nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        params_dict[(log_nu, T)] = log_theta

    print(f'n_samples={len(params_dict)}')

    # designate demographic model, sample size, and extrapolation grid
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]
    # theta_list = [1]
    # specify param in log scale
    logs = [True, False]
    # both N and t should be in log scale, but T is kept in non-log
    # scale here for fs generating step with dadi

    # generate data with dadi (parallelized function from data_manip)
    data = generating_data(
        list(params_dict.keys()), theta_list, func, ns, pts_l, logs)
    # data is a list of dict with one single element since theta_list=[1]

    # Make new params in non-dadi unit (nu and T become N and t)
    # make new dict to store new param labels and corresponding fs
    new_data = []

    # right now we have two dicts: params_dict and data[0],
    # (use data[0], not data: generating_data() created list of dicts)
    # both have n_samples items and the same (log_nu, T) tuple as keys

    # iterate through the keys to make new labels for data_dict
    for each in data:
        data_dict = {}
        for key in list(params_dict.keys()):
            # get theta in log and convert to non-log
            log_theta = params_dict[key]
            theta = 10**log_theta

            # convert nu and T to N and t using non-log theta
            nu = 10**key[0]  # key[0] is nu in log scale
            N = math.log10(nu * theta / 4)
            t = math.log10(key[1] * theta / 2)  # key[1] is T
            # both N and t will be in log scale
            new_key = N, t

            # make new fs using theta
            fs = each[key]
            scaled_fs = fs * theta

            # store labels and data into data_dict
            data_dict[new_key] = scaled_fs
        new_data.append(data_dict)

    return new_data


if __name__ == "__main__":
    # train_data_1000 = generate_data_1d_2epoch_unnorm(1000)
    # pickle.dump(train_data_1000, open(f'data/train_data_1000', 'wb'), 2)
    
    # train_data_5000 = generate_data_1d_2epoch_unnorm(5000)
    # pickle.dump(train_data_1000, open(f'data/train_data_5000', 'wb'), 2)
    
    # train_data_10000 = generate_data_1d_2epoch_unnorm(10000)
    # pickle.dump(train_data_1000, open(f'data/train_data_10000', 'wb'), 2)

    # test_data = generate_data_1d_2epoch_unnorm(100)
    # pickle.dump(test_data, open(f'data/test_data', 'wb'), 2)

    test_data_full = generate_data_1d_2epoch_unnorm(100, [1, 10000, 1000, 100])
    pickle.dump(test_data_full, open(f'data/test_data_full', 'wb'), 2)
