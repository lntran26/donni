import util
import dadi
import pickle
import random
from copy import deepcopy
import numpy as np
import time

if __name__ == '__main__': 
    # import test sets previously generated for 2D-migration
    list_test_dict = pickle.load(open('../../data/2d-splitmig/test-data-corrected','rb'))
    # randomly select 5 datasets from each variance case
    # be careful to pick different params key because
    # each variance list share the same set of params
    # which may result in fewer datasets picked after loops
    test_data = {}
    for test_dict in list_test_dict:
        for i in range(5):
            params, fs = random.choice(list(test_dict.items()))
            # pick a different set until find a unique param value
            while params in test_data: 
                params, fs = random.choice(list(test_dict.items()))
            # once find a unique params, escape while loop and add that set 
            # to test_data dict
            test_data[params] = fs

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
    # format:list_pred=[[20 np.arrays of 20 pred_param;theta1],[],[],[theta10k]]


    # benchmarking
    # an idea: test if use suggestion from just the theta=1 rfr
    # vs using suggestion from an average of all 4 rfr guess
    
    # common process: set up dadi
    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    # func = split_mig_fixed_theta
    ns = [20,20]
    pts_l = [40, 50, 60]
    func_ex = dadi.Numerics.make_extrap_func(func)
    lower_bound, upper_bound = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]
    
    # THREE DIFFERENT STARTING POINTS
    # # for each set of test data set: load fs
    # for p in test_data:
    #     fs = test_data[p]

    for i in range(2):
        print("DATA SET #", i+1)
        # input each fs from test_data to infer from
        p = list_key[0][i] # true params
        fs = test_data[p]
        p_transform = [10**p[0], 10**p[1], p[2], p[3]]
        print("True params:", [round(val, 5) for val in p_transform])
        print("True Max Log Likelihood:", round(dadi.Inference.ll(fs, fs),5))

        # 1. Generic starting point for regular dadi procedure
        sel_params_generic = [1, 1, 0.95, 4.5] # mid point of range
        print("Selected generic params:", sel_params_generic)
        
        # perturb param
        p0 = dadi.Misc.perturb_params(sel_params_generic, 
                            lower_bound=lower_bound, upper_bound=upper_bound)
        
        # run optimization & print results
        start = time.time()
        popt, llnlopt = dadi.Inference.opt(p0, fs, func_ex, pts_l, 
                                        lower_bound=lower_bound,
                                        upper_bound=upper_bound,
                                        multinom=True,
                                        verbose=5)
        print('Optimization time for dadi: {0:.2f}s'.format(time.time()-start))
        print("Optimized params:",[round(p,5) for p in popt])
        print("Max Log Likelihood:", round(llnlopt,5))
        print('\n',"####################",'\n')

       
        # 2. Starting with Prediction from rfr trained on theta=1
        sel_params_RFR_1 = list_pred[0][i]
        print("Selected params from RFR trained on theta=1:", 
                    [round(p,5) for p in sel_params_RFR_1])
        
        # perturb param
        p0 = dadi.Misc.perturb_params(sel_params_RFR_1, 
                            lower_bound=lower_bound, upper_bound=upper_bound)
        
        # run optimization & print results
        start = time.time()
        popt, llnlopt = dadi.Inference.opt(p0, fs, func_ex, pts_l, 
                                        lower_bound=lower_bound,
                                        upper_bound=upper_bound,
                                        multinom=True,
                                        verbose=5)
        print('Optimization time for dadi + RFR_1: {0:.2f}s'.format(time.time()-start))
        print("Optimized params:",[round(p,5) for p in popt])
        print("Max Log Likelihood:", round(llnlopt,5))
        print('\n',"####################",'\n')

        # 3. Average prediction from all four RFR trained on different thetas    
        arr = np.array([list_pred[0][i],list_pred[1][i],
                            list_pred[2][i],list_pred[3][i]])
        sel_params_RFR_avg = np.mean(arr, axis=0).tolist()
        print("Selected params from average of 4 RFRs:", 
                    [round(p,5) for p in sel_params_RFR_avg])
        
        # perturb param
        p0 = dadi.Misc.perturb_params(sel_params_RFR_avg, 
                            lower_bound=lower_bound, upper_bound=upper_bound)
        
        # run optimization & print results
        start = time.time()
        popt, llnlopt = dadi.Inference.opt(p0, fs, func_ex, pts_l, 
                                        lower_bound=lower_bound,
                                        upper_bound=upper_bound,
                                        multinom=True,
                                        verbose=5)
        print('Optimization time for dadi + RFR_avg: {0:.2f}s'.format(time.time()-start))
        print("Optimized params:",[round(p,5) for p in popt])
        print("Max Log Likelihood:", round(llnlopt,5))
        print('\n',"####################",'\n')
        print('\n')