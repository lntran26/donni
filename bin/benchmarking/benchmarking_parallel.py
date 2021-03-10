import util
import dadi
import pickle
import random
from copy import deepcopy
import numpy as np
import time
from multiprocessing import Pool

# multiprocessing Pools can't have newly-created functions passed to them. 
# So we need to do the func_ex inside it here.
def worker_func_opt(args):
    (p, fs, func, pts_l, lb, ub) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    # perturb param
    # p0 = dadi.Misc.perturb_params(p, lower_bound=lb, upper_bound=ub)
    return dadi.Inference.opt(p, fs, func_ex, pts_l, lower_bound=lb,
                                    upper_bound=ub)

def dadi_opt_parallel(p_true_list, p_list, fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for dadi_optimization using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [(p, fs, func, pts_l, lb, ub) for p, fs in zip(p_list, fs_list)]
    with Pool(processes=ncpu) as pool:
            opt_list = pool.map(worker_func_opt, args_list)
    
    # for p_true, opt in zip(p_true_list, opt_list):
    #     print("True params:", [round(val,5) for val in p_true])
    #     print("True Max Log Likelihood:", 
    #                 round(dadi.Inference.ll_multinom(fs, fs),5))
    #     print("Optimized params:",[round(p,5) for p in opt[0]])
    #     print("Max Log Likelihood:", round(opt[1],5), '\n')

    return opt_list

if __name__ == '__main__': 
    test_set = pickle.load(open('benchmarking_test_set','rb'))
    p_true_list = [test_set[i][0] for i in range(20)]
    fs_list = [test_set[i][1] for i in range(20)]
    p1_list = [test_set[i][2] for i in range(20)]
    p2_list = [test_set[i][2] for i in range(20, 40)]
    p3_list = [test_set[i][2] for i in range(40, 60)]

    # benchmarking: parallel
    # common process: set up dadi
    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]

    # THREE DIFFERENT STARTING POINTS
    # 1. Generic starting point for regular dadi procedure
    p1_opt = dadi_opt_parallel(p_true_list, p1_list, fs_list, func, pts_l, lb, ub)
    
    # 2. Starting with Prediction from rfr trained on theta=1
    p2_opt = dadi_opt_parallel(p_true_list, p2_list, fs_list, func, pts_l, lb, ub)

    # 3. Average prediction from all four RFR trained on different thetas    
    p3_opt = dadi_opt_parallel(p_true_list, p3_list, fs_list, func, pts_l, lb, ub)

    results_list = []
    for i in range(20):
        results_list.append((p_true_list[i], p1_opt[i], p2_opt[i], p3_opt[i]))

    pickle.dump(results_list, open('benchmarking_results_set', 'wb'), 2)