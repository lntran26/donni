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
    
 
if __name__ == '__main__': 
    
    print("---------START---------\n")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    test_set = pickle.load(open('test-data-corrected-2','rb'))[2]
    p_true = list(test_set.keys())[:50] # log scale
    fs_data = [1000*test_set[p] for p in p_true]
    # convert to non-log scale
    p_true = [[10**p[0], 10**p[1], p[2], p[3]] for p in p_true]

    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]
    
    NUM_P = 10

    # make ten random starting points 
    p = []
    for i in range(NUM_P):
        nu1 = 10 **(random.random() * 4 - 2)
        nu2 = 10**(random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        m = random.random() * 9 + 1
        p.append([nu1, nu2, T, m])
        
    '''     
    # starting points
    p1 = [10**-1.5, 10**1, 0.5, 2]
    p2 = [10**1, 10**-1.5, 1.5, 6]
    p3 = [10**1.2, 10**1.2, 1, 8]
    p4 = [10**-0.5, 10**0.5, 1.8, 3]
    p5 = [10**0.9, 10**0.9, 1, 4]
    '''
    
    p_list = []
    for i in range(len(p_true)):
        for j in range(len(p)):
            p_list.append(p[j]) # append each param once 50 times
    
    fs_list = []
    for fs in fs_data:
        for i in range(len(p)):
            fs_list.append(fs) # append each fs 10 times
    
    # p_list is of the form [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p1...]
    # fs_list is of the form [fs1, fs1, fs1, fs1, ... fs1 (10 times), fs2, fs2...]
    # both are length 500

    popt_list = dadi_opt_parallel(p_list, fs_list, func, pts_l, lb, ub)
    
    # transform this into 50 lists of length 10
    popt_list = [popt_list[i:i + len(p)] for i in range(0, len(popt_list), len(p))]
    popt_sorted = []
    # sort and print all of the sets of optimizations
    for i in range(len(popt_list)):
        #print(f"Param Set #{i+1}")
        #print(f"True:\t\t\t\t\t{p_true[i][0], p_true[i][1], p_true[i][2], p_true[i][3]}")
        lst = popt_list[i]
        zipped = zip(p, lst)
        zipped = sorted(zipped, key=lambda x:-x[1][1])
        popt_sorted.append(zipped)
        '''
        print("------Start, Opt, LL------")
        for j in range(len(zipped)):
            p_start = zipped[j][0]
            popt = zipped[j][1]
            print(f"{p_start}:\t\t\t{popt[0]}\t\t\t{popt[1]}")
        '''
        
    '''
    for i in range(50):
        print(f"True:\t{10**p_true[i][0], 10**p_true[i][1], p_true[i][2], p_true[i][3]}")
        print("Start\tOpt\tLL")
        print(f"{p1}:\t{popt_list[i*5][0]}\t{popt_list[i*5][1]}")
        print(f"{p2}:\t{popt_list[i*5+1][0]}\t{popt_list[i*5+1][1]}")
        print(f"{p3}:\t{popt_list[i*5+2][0]}\t{popt_list[i*5+2][1]}")
        print(f"{p4}:\t{popt_list[i*5+3][0]}\t{popt_list[i*5+3][1]}")
        print(f"{p5}:\t{popt_list[i*5+4][0]}\t{popt_list[i*5+4][1]}")
    '''
    pickle.dump(popt_sorted, open(f'dadi_opt_results_pre_converge', 'wb'), 2)
    
    print("\n----------END----------")
    
        
    
