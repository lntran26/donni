'''
Method for generating dadi-simulated fs datasets
'''
import sys
import random
from multiprocessing import Pool
import dadi
from scipy.stats import loguniform
import math
import numpy as np


def worker_func(args: tuple):
    '''
    Helper function for generate_fs() to perform parallelization with Pool
    Return: a single fs
    '''

    (p, func, ns, pts_l, folded) = args
    if not folded:
        func = dadi.Numerics.make_anc_state_misid_func(func)
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)


def generate_fs(func, params_list, logs, theta, ns, pts_l,
                norm=True, sampling=True, folded=False,
                bootstrap=False, n_bstr=200, ncpu=None):
    '''
    Parallelized generation of a dataset of multiple fs based on an input
    demographic model and a list of several demographic parameters
    Inputs:
        func: dadi demographic model
        params_list: demographic model param sets
        logs: indicate which dem param is in log10 values
        theta: value of theta
        ns: population sample size(s)
        pts_l: dadi extrapolation grid values
        norm: whether to sample from and normalize the fs
        folded: whether to fold the fs
        bootstrap: whether to generate bootstrap data
        n_bstr: number of bootstrap fs per original fs
        n_cpu: integer num of CPUs to use for generating data
            (None means using all)
    Output: dataset dictionary with format params:fs
    '''

    arg_list = []
    new_params_list = []
    for p in params_list:
        delog_p = [10**p[i] if logs[i] else p[i] for i in range(len(logs))]
        if not folded:
            misid = random.random() / 4  # range 0 to .25
            delog_p.append(misid)
            new_p = list(p)
            new_p.append(misid)
            new_params_list.append(new_p)
        arg_list.append((delog_p, func, ns, pts_l, folded))
    if not folded:
        params_list = new_params_list

    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(worker_func, arg_list)

    data_dict = {}
    qual_check = []
    for params, fs in zip(params_list, fs_list):
        params = tuple(params)
        # assign zeros to masked entries of fs
        fs.flat[0] = 0
        fs.flat[-1] = 0
        # quality check each fs: store the number of entries in the fs
        # that is negative, nan, or infinity
        num_neg = (fs < 0).sum()
        num_nan = (np.isnan(fs)).sum()
        num_inf = (np.isposinf(fs)).sum()
        fs_qual = [num_neg, num_nan, num_inf]
        # store more detailed stats for negative entries
        if np.any(fs < 0):
            sum_neg = np.sum(fs[fs < 0])
            fs_qual += [fs.min(), sum_neg, fs.sum(), abs(sum_neg/fs.sum())]
        else:
            fs_qual += [0, 0, 0, 0]
        # append stat for each fs to a list of all fs stats
        qual_check.append(fs_qual)
        # convert any negative entry in fs before further processing
        fs = abs(fs)

        # generate data for bootstrapping
        if bootstrap:
            if theta == 1:
                sys.exit("Cannot bootstrap fs with theta=1")
            fs_tostore = (theta*fs).sample()
            data_dict[params] = [fs_tostore, []]
            for _ in range(n_bstr):  # num bootstrap samples for each fs
                data_dict[params][1].append(fs_tostore.sample())
        # generate regular data
        else:
            fs_tostore = theta*fs
            # sampling step, skip if theta==1
            if sampling and theta != 1:
                fs_tostore = fs_tostore.sample()
                # rerun sampling step if fs.sum() is zero
                while fs_tostore.sum() == 0:
                    fs_tostore = theta*fs.sample()
            # normalization step
            if norm:
                fs_tostore = fs_tostore/fs_tostore.sum()
            # fold fs
            if folded:
                fs_tostore = fs_tostore.fold()
            data_dict[params] = fs_tostore
    return data_dict, qual_check


def _get_subsequent_layer(init_hls: list, n_sub_layer: int):
    """
    Helper method for calculating subsequent layer size
    base on the initial hls list and number of subsequent layers wanted
    Return: full list of different hidden_layer_size options
    """
    # stop recursion condition
    if n_sub_layer == 0:
        return init_hls

    # get the last layer size
    init_layer_size = init_hls[-1][-1]
    # reduce init layer size to get next layer size
    next_layer_size = int(init_layer_size / 2)
    # make new tuple with new layer size and append to init_hls
    new_hls = list(init_hls[-1])
    new_hls.append(next_layer_size)
    init_hls.append(tuple(new_hls))

    return _get_subsequent_layer(init_hls, n_sub_layer - 1)


def get_hyperparam_tune_dict(sample_sizes: list):
    """"
    Method for generating hyperparam dictionary for tuning
    based on input data size
    Return: a dict specifying hyperparam options for tuning
    """
    # calculate input layer size
    n_input = math.prod(sample_sizes)

    # different tiers of sample size to determine hyperparam dict spec
    if n_input < 100:
        # for: 10,20,40
        first_layer_size = int(n_input / 2)
        hls_list = _get_subsequent_layer([(first_layer_size,)], 1)
    elif 80 <= n_input <= 100:
        # for: 80 and 10**2=100
        first_layer_size = int(n_input / 2)
        hls_list = _get_subsequent_layer([(first_layer_size,)], 2)
    elif 100 <= n_input < 200:
        # for 160
        hls_list = _get_subsequent_layer([(50,)], 2)
    else:
        hls_list = _get_subsequent_layer([(50,)], 2)
        hls_list += _get_subsequent_layer([(25,)], 2)

    # put together hyperparam dict for tuning
    tune_dict = {'hidden_layer_sizes': hls_list,
                 'activation': ['relu'],
                 'solver': ['lbfgs', 'adam'],
                 'alpha': loguniform(1e-5, 1e-1),
                 'learning_rate_init': loguniform(1e-5, 1e-1)}
    return tune_dict
