from multiprocessing import Pool
import dadi
import numpy as np


def worker_func(args):
    '''Helper function for generating_data()
    Help with parallelization with Pool'''
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)


# note: command-line: params_list will be made from specification
# of number of params to generation, and upper bound and lower bound vals
# ul and ll have to be specified for each dem param
# or maybe rule will be included and specified for each dem model?
def generate_data(theta_list, params_list, func, ns, pts_l,
                  logs=None, sample=True, norm=True,
                  bootstrap=False, n_bstr=200, ncpu=None):
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
                for i in range(n_bstr):  # num bootstrap samples for each fs
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


if __name__ == "__main__":
    # designate dadi demographic model, sample size, and extrapolation grid
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]
    # specify param in log scale
    logs = [True, False]

    # generate params
    params_list = []
    while len(params_list) < 100:
        # pick random values in specified range
        # nu range: 0.01-100; T range: 0.1-2
        log_nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        # if T/10**log_nu <= 5: # exclude certain T & nu combo
        params_list.append((log_nu, T))

    # generate data
    data = generate_data(theta_list, params_list, func, ns, pts_l, logs)

    # save data
    pickle.dump(data, open(
        '../tests/test_data/1d_2epoch_100fs_exclude', 'wb'), 2)
