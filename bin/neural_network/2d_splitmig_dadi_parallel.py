import util
import dadi
import pickle
import random
from copy import deepcopy
import numpy as np
import time
from multiprocessing import Pool

def worker_func_opt(args):
    (p, fs, func, pts_l, lb, ub) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    # perturb param
    # p0 = dadi.Misc.perturb_params(p, lower_bound=lb, upper_bound=ub)
    return dadi.Inference.opt(p, fs, func_ex, pts_l, lower_bound=lb,
                                    upper_bound=ub)

def dadi_opt_parallel(fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for dadi_optimization using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [([10**0.8, 10**1.2, 0.8, 3.5], fs, func, pts_l, lb, ub) for fs in fs_list]
    with Pool(processes=ncpu) as pool:
            opt_list = pool.map(worker_func_opt, args_list)
    return opt_list

if __name__ == '__main__': 
    test_set = pickle.load(open('test_set','rb'))
    p_true_1 = test_set[0].keys()
    fs_list_1 = test_set[0].values()
    
    p_true_2 = test_set[1].keys()
    fs_list_2 = test_set[1].values()
    
    p_true_3 = test_set[2].keys()
    fs_list_3 = test_set[2].values()
    
    p_true_4 = test_set[3].keys()
    fs_list_4 = test_set[3].values()
    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]

    results_list = []

    p1_opt = dadi_opt_parallel(fs_list_1, func, pts_l, lb, ub) # theta = 1
    p2_opt = dadi_opt_parallel(fs_list_2, func, pts_l, lb, ub) # theta = 100
    p3_opt = dadi_opt_parallel(fs_list_3, func, pts_l, lb, ub) # theta = 1000
    p4_opt = dadi_opt_parallel(fs_list_4, func, pts_l, lb, ub) # theta = 10000

    for lst in (p1_opt, p2_opt, p3_opt, p4_opt):
        results_list.append(lst)

    pickle.dump(results_list, open('dadi_parallel_results', 'wb'), 2)
    print("END")