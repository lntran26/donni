'''
Method for generating dadi-simulated fs datasets
'''
from multiprocessing import Pool
import dadi


def worker_func(args):
    '''Helper function for generating_data()
    Help with parallelization with Pool'''
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)


def generate_fs(func, params_list, logs, theta_list, ns, pts_l,
                norm=True, bootstrap=False, n_bstr=200, ncpu=None):
    '''



    Parallelized version for generating data using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine.
    Otherwise user can specify a limit (int).
    logs is a list where len(logs) = len(p) and
    logs[i] = True if p[i] is in log scale.
    If bootstrap is True, will generate data for bootstrapping purpose.
    Returns a list of dictionaries where each dictionary stores
    a data set for training or testing RFR algorithm.
    Data dictionaries have structure params:fs.
    '''
    arg_list = [([10**p[i] if logs[i] else p[i] for i in range(len(logs))],
                 func, ns, pts_l) for p in params_list]

    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(worker_func, arg_list)


    # data_dict = {}
    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            fs.flat[0] = 0
            fs.flat[-1] = 0
            if bootstrap:  # generate data for bootstrapping
                fs_tostore = (theta*abs(fs)).sample()
                data_dict[params] = [fs_tostore, []]
                for i in range(n_bstr):  # num bootstrap samples for each fs
                    data_dict[params][1].append(fs_tostore.sample())

            else:  # generate regular data
                if theta == 1:
                    fs_tostore = fs
                else: # theta != 1 for noisy or non-normalized data
                    if norm: # sample and normalize
                        fs_tostore = (theta*abs(fs).sample())
                        if fs_tostore.sum() == 0:  # check zero fs after sampling
                            pass
                        else:
                            fs_tostore = fs_tostore/fs_tostore.sum()
                    else: # no sampling and normalizing
                        fs_tostore = (theta*abs(fs))
            data_dict[params] = fs_tostore


                # if theta == 1:
                #     fs_tostore = fs
                # elif sample:
                #     fs_tostore = (theta*abs(fs)).sample()
                # else:
                #     fs_tostore = (theta*abs(fs))
                # if fs_tostore.sum() == 0:  # check non-zero fs
                #     pass
                # elif norm:
                #     data_dict[params] = fs_tostore/fs_tostore.sum()
                # else:
                #     data_dict[params] = fs_tostore
        list_dicts.append(data_dict)
    # return data_dict
    return list_dicts
