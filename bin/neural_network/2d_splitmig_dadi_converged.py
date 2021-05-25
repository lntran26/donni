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

def dadi_opt_parallel(p_list, fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for dadi_optimization using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [(p, fs, func, pts_l, lb, ub) for p,fs in zip(p_list, fs_list)]
    with Pool(processes=ncpu) as pool:
            opt_list = pool.map(worker_func_opt, args_list)
    return opt_list
    
def converged(top):
    if round(abs(top[0][1][1]-top[1][1][1]), 2) <= 0.05:
        if round(abs(top[0][1][1]-top[2][1][1]), 2) <= 0.05:
            return True
    return False

if __name__ == '__main__': 
    test_set = pickle.load(open('test-data-corrected-2','rb'))[2]
    p_true = list(test_set.keys())[:50] # log scale
    fs_data = [1000*test_set[p] for p in p_true]
    # convert to non-log scale
    p_true = [[10**p[0], 10**p[1], p[2], p[3]] for p in p_true]
    
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]
    
    true_pred = []
    popt_sorted = pickle.load(open('dadi_opt_results_pre_converge','rb'))
    for i in range(len(popt_sorted)):
        lst = popt_sorted[i]
        print(f"Param Set #{i+1}")
        print(f"True:\t\t\t\t\t{p_true[i][0], p_true[i][1], p_true[i][2], p_true[i][3]}")
        if (converged(lst)):
            true_pred.append([p_true[i], lst[0][1][0]])
            for j in range(3):
                print(lst[j][1])
        else:
            p0_list = [dadi.Misc.perturb_params(lst[0][1][0], lower_bound=lb, upper_bound=ub) for i in range(3)]
            popt_list_2 = dadi_opt_parallel(p0_list, [fs_data[i]]*3, func, pts_l, lb, ub)
            popt_zipped = zip(p0_list, popt_list_2)
            popt_zipped = sorted(popt_zipped, key=lambda x:-x[1][1])
            if (converged(popt_zipped)):
                true_pred.append([p_true[i], popt_zipped[0][1][0]])
                for j in range(3):
                    print(popt_zipped[j][1])
            else:
                true_pred.append([p_true[i], None])
                print("Not converged")
        print()
        
    pickle.dump(true_pred, open(f'dadi_opt_results_converged', 'wb'), 2)
    
    
    
    