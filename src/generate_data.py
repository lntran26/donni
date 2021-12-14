from multiprocessing import Pool
import dadi
import numpy as np


def worker_func(args):
    '''Helper function for generating_data()
    Help with parallelization with Pool'''
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)


def generating_data(params_list, theta_list, func, ns, pts_l, logs=None,
                    sample=True, norm=True, bootstrap=False, ncpu=None):
    '''Parallelized version for generating data using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    logs is a list where len(logs) = len(p) and 
    logs[i] = True if p[i] is in log scale. 
    If bootstrap is True, will generate data for bootstrapping purpose.
    Returns a list of dictionaries where each dictionary stores
    a data set for training or testing RFR algorithm. 
    Dictionaries have structure params:fs.
    '''
    if logs != None:
        arg_list = [([10**p[i] if logs[i] else p[i] for i in range(len(logs))],
                    func, ns, pts_l) for p in params_list]
    else:
        arg_list = [(p, func, ns, pts_l) for p in params_list]

    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(worker_func, arg_list)

    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            if bootstrap:  # only run this if want to generate bootstrap data sets
                fs_tostore = (theta*abs(fs)).sample()
                data_dict[params] = [fs_tostore, []]
                for i in range(200):  # 200 bootstrap samples for each fs
                    data_dict[params][1].append(fs_tostore.sample())
            else:  # generate data for ML training and testing
                fs.flat[0] = 0
                fs.flat[-1] = 0
                if theta == 1:
                    fs_tostore = fs
                elif sample:
                    fs_tostore = (theta*abs(fs)).sample()
                else:
                    fs_tostore = (theta*abs(fs))
                if fs_tostore.sum() == 0:
                    pass
                elif norm:
                    data_dict[params] = fs_tostore/fs_tostore.sum()
                else:
                    data_dict[params] = fs_tostore
        list_dicts.append(data_dict)
    return list_dicts
