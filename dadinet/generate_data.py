'''
Method for generating dadi-simulated fs datasets
'''
import sys
from multiprocessing import Pool
import dadi


def worker_func(args: tuple):
    '''
    Helper function for generate_fs() to perform parallelization with Pool
    '''

    (p, func, ns, pts_l) = args
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

    arg_list = [([10**p[i] if logs[i] else p[i] for i in range(len(logs))],
                 func, ns, pts_l) for p in params_list]

    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(worker_func, arg_list)

    data_dict = {}
    for params, fs in zip(params_list, fs_list):
        # assign zeros to masked entries of fs
        fs.flat[0] = 0
        fs.flat[-1] = 0

        # generate data for bootstrapping
        if bootstrap:
            if theta == 1:
                sys.exit("Cannot bootstrap fs with theta=1")
            fs_tostore = (theta*abs(fs)).sample()
            data_dict[params] = [fs_tostore, []]
            for _ in range(n_bstr):  # num bootstrap samples for each fs
                data_dict[params][1].append(fs_tostore.sample())

        # generate regular data
        else:
            fs_tostore = theta*abs(fs)
            # sampling step, skip if theta==1
            if sampling and theta != 1:
                fs_tostore = fs_tostore.sample()
                # rerun sampling step if fs.sum() is zero
                while fs_tostore.sum() == 0:
                    fs_tostore = theta*abs(fs).sample()
            # normalization step
            if norm:
                fs_tostore = fs_tostore/fs_tostore.sum()
            # fold fs
            if folded:
                fs_tostore = fs_tostore.fold()
            data_dict[params] = fs_tostore

    return data_dict
