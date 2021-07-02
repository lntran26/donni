# full optimization process

import util
import dadi
import pickle
import random
import numpy as np
import time
from multiprocessing import Pool

def worker_func_opt(args):
    (p, fs, func, pts_l, lb, ub) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    p0 = dadi.Misc.perturb_params(p, lower_bound=lb, upper_bound=ub)
    popt, ll = dadi.Inference.opt(p0, fs, func_ex, pts_l, lower_bound=lb,
                                    upper_bound=ub)
    max_ll = ll
    i = 0
    # max of five optimizations
    while i < 5 and ll >= max_ll:
        max_ll = ll
        p = popt
        p0 = dadi.Misc.perturb_params(p, lower_bound=lb, upper_bound=ub)
        popt, ll = dadi.Inference.opt(p0, fs, func_ex, pts_l, lower_bound=lb,
                                    upper_bound=ub)
        i += 1
    return p, max_ll, i

def dadi_opt_parallel(p_list, fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for dadi_optimization using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [(p, fs, func, pts_l, lb, ub) for p,fs in zip(p_list, fs_list)]
    with Pool(processes=ncpu) as pool:
            opt_list = pool.map(worker_func_opt, args_list)
    return opt_list
    
 
if __name__ == '__main__': 
    
    print("---------START---------\n")
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # all length 50, indices should match (e.g., p_true_list[0] was used to generate fs_list[0])
    p_true_list = pickle.load(open('dadi_p_true','rb'))
    fs_list = pickle.load(open('dadi_fs_data','rb'))
    p_start_list = pickle.load(open('dadi_start','rb'))

    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]

    popt_list = dadi_opt_parallel(p_start_list, fs_list, func, pts_l, lb, ub)
    
    for i in range(len(popt_list)):
        print(f"True: {10**p_true_list[i][0], 10**p_true_list[i][1], p_true_list[i][2], p_true_list[i][3]}")
        print(f"Opt : {popt_list[i][0]}\t\t\tLL: {popt_list[i][1]}")
        print(f"Num opts: {popt_list[i][2]}\n")

    pickle.dump(popt_list, open(f'../../results/{timestr}_dadi_opt_full_results', 'wb'), 2)
    
    print("\n----------END----------")
    
        
    

