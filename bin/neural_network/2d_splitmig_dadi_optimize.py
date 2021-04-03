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

def dadi_opt_parallel(p, fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for dadi_optimization using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [(p, fs, func, pts_l, lb, ub) for fs in fs_list]
    with Pool(processes=ncpu) as pool:
            opt_list = pool.map(worker_func_opt, args_list)
    return opt_list
    

if __name__ == '__main__': 
    test_set = pickle.load(open('benchmarking_test_set_1','rb'))
    p_true = [test_set[i][0] for i in range(20)]
    fs = [test_set[i][1] for i in range(20)]

    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]

    p1 = [10**-1.5, 10**1, 0.5, 2]
    p2 = [10**1, 10**-1.5, 1.5, 6]
    p3 = [10**1.2, 10**1.2, 1, 8]
    p4 = [10**-0.5, 10**0.5, 1.8, 3]
    p5 = [10**0.9, 10**0.9, 1, 4]
    
    popt_1 = dadi_opt_parallel(p1, fs, func, pts_l, lb, ub)
    popt_2 = dadi_opt_parallel(p2, fs, func, pts_l, lb, ub)
    popt_3 = dadi_opt_parallel(p3, fs, func, pts_l, lb, ub)
    popt_4 = dadi_opt_parallel(p4, fs, func, pts_l, lb, ub)
    popt_5 = dadi_opt_parallel(p5, fs, func, pts_l, lb, ub)
    
    opt_results = [popt_1, popt_2, popt_3, popt_4, popt_5]
    pickle.dump(opt_results, open('dadi_opt_results', 'wb'), 2)

    for i in range(len(p_true)):
        print(f"True:\t{p_true[i]}")
        print("Start\tOpt\tLL")
        print(f"{p1}:\t{popt_1[i][0]}\t{popt_1[i][1]}")
        print(f"{p2}:\t{popt_2[i][0]}\t{popt_2[i][1]}")
        print(f"{p3}:\t{popt_3[i][0]}\t{popt_3[i][1]}")
        print(f"{p4}:\t{popt_4[i][0]}\t{popt_4[i][1]}")
        print(f"{p5}:\t{popt_5[i][0]}\t{popt_5[i][1]}")

    
        
    