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
    p0 = dadi.Misc.perturb_params(p, lower_bound=lb, upper_bound=ub)
    return dadi.Inference.opt(p0, fs, func_ex, pts_l, lower_bound=lb,
                                    upper_bound=ub)

def dadi_opt_parallel(p_true_list, p_list, fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for dadi_optimization using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [(p, fs, func, pts_l, lb, ub) for p, fs in zip(p_list, fs_list)]
    with Pool(processes=ncpu) as pool:
            opt_list = pool.map(worker_func_opt, args_list)
    
    for p_true, opt in zip(p_true_list, opt_list):
        print("True params:", [round(val,5) for val in p_true])
        print("True Max Log Likelihood:", 
                    round(dadi.Inference.ll_multinom(fs, fs),5))
        print("Optimized params:",[round(p,5) for p in opt[0]])
        print("Max Log Likelihood:", round(opt[1],5), '\n')

if __name__ == '__main__': 
    # import test sets previously generated for 2D-migration
    list_test_dict = pickle.load(open('../../data/2d-splitmig/test-data-corrected','rb'))
    # randomly select 5 datasets from each variance case
    test_data = {}
    for test_dict in list_test_dict:
        for i in range(5):
            params, fs = random.choice(list(test_dict.items()))
            # pick a different set until find a unique param value
            while params in test_data: 
                params, fs = random.choice(list(test_dict.items()))
            # once find a unique params, escape while loop and add that set 
            # to test_data dict
            # test_data[params] = fs
            # more realistic data option--> scale fs by 1000
            test_data[params] = fs*1000

    # import trained RFR
    list_rfr = pickle.load(open('../../data/2d-splitmig/list-rfr','rb'))
    #  use rfr to give predictions for test_data
    list_pred = []
    list_key = [] # list of key (param) to get fs from test_data dict
    for rfr in list_rfr:
        y_true, y_pred = util.rfr_test(rfr, test_data)
        # perform log transform on predictionr results and 
        # convert p in y_pred from np.array to list format
        y_pred_transform = []
        for p in y_pred:
            y_pred_transform.append([10**p[0], 10**p[1], p[2], p[3]])
        list_pred.append(y_pred_transform)
        # also copy y_true to get keys
        list_key.append(deepcopy(y_true))

    # benchmarking: parallel
    
    # common process: set up dadi
    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    lb = [1e-2,1e-2,0.1,1]
    ub = [1e2,1e2,2,10]
    
    fs_list, p_true_list = [], []
    p1_list, p2_list, p3_list = [],[],[]
    
    for i in range(2): # change to increase # of test case run
        # input each fs from test_data to infer from
        p = list_key[0][i] # true params
        p_transform = [10**p[0], 10**p[1], p[2], p[3]]
        p_true_list.append(p_transform)

        fs = test_data[p]
        fs_list.append(fs)
        
        # List of generic starting points
        p1_list.append([1, 1, 0.95, 4.5])

        # List of starting points from RFR theta 1 predictions
        sel_params_RFR_1 = list_pred[0][i]
        p2_list.append(sel_params_RFR_1)

        # List of starting points from average 4 RFRs prediction
        arr = np.array([list_pred[0][i],list_pred[1][i],
                        list_pred[2][i],list_pred[3][i]])
        sel_params_RFR_avg = np.mean(arr, axis=0).tolist()
        p3_list.append(sel_params_RFR_avg)


    # THREE DIFFERENT STARTING POINTS
    # 1. Generic starting point for regular dadi procedure
    print('\n',"####################",'\n')
    print("CASE 1: dadi only")
    start = time.time()
    dadi_opt_parallel(p_true_list, p1_list, fs_list, func, pts_l, lb, ub)
    print('Optimization time for dadi: {0:.2f}s'.format(time.time()-start))
    
    # 2. Starting with Prediction from rfr trained on theta=1
    print("CASE 2: dadi + RFR_1")
    start = time.time()
    dadi_opt_parallel(p_true_list, p2_list, fs_list, func, pts_l, lb, ub)
    print('Optimization time for dadi + RFR_1: {0:.2f}s'.format(time.time()-start))

    # 3. Average prediction from all four RFR trained on different thetas    
    print("CASE 3: dadi + 4_RFR_avg")
    start = time.time()
    dadi_opt_parallel(p_true_list, p3_list, fs_list, func, pts_l, lb, ub)
    print('Optimization time for dadi + 4_RFR_avg: {0:.2f}s'.format(time.time()-start))