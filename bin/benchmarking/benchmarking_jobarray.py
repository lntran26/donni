import util
import dadi
import pickle
import random
from copy import deepcopy
import numpy as np
import time
import sys,os

if __name__ == '__main__': 
    # import test sets previously generated for 2D-split-migration
    list_test_dict = pickle.load(open(
        '/groups/rgutenk/lnt/projects/ml-dadi/data/test-data-corrected','rb'))
    # randomly select 5 datasets from each variance case
    test_data = {}
    for test_dict in list_test_dict:
        for i in range(5):
            params, fs = random.choice(list(test_dict.items()))
            # pick a different set until find a unique param value
            while params in test_data: 
                params, fs = random.choice(list(test_dict.items()))
            # once find a unique params, escape while loop and add that set 
            # scale by theta=1000 for a more realistic fs
            test_data[params] = fs*1000

    # import trained RFR
    list_rfr = pickle.load(open(
                '/groups/rgutenk/lnt/projects/ml-dadi/data/list-rfr','rb'))
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
    # common process: set up dadi
    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    func_ex = dadi.Numerics.make_extrap_func(func)
    lb, ub = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]

    # 1 list of fs, 1 list of p_true
    fs_list, p_true_list = [], []
    # 3 lists of different starting point p0
    p1_list, p2_list, p3_list = [],[],[]
    
    for i in range(20):
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

    # to make it easier for job array submission
    # append all three lists together into one list of all starting points
    p0_list = p1_list + p2_list + p3_list
    # extend fs_list and p_true_list 2 times to match length
    # fs_list_ext = fs_list * 3
    # p_true_list_ext = p_true_list * 3

    if 'SLURM_SUBMIT_DIR' in os.environ:
    # Set module search path so it will search in qsub directory
        sys.path.insert(0, os.environ['SLURM_SUBMIT_DIR'])
    # Set current working directory to qsub directory
        os.chdir(os.environ['SLURM_SUBMIT_DIR'])
    
    # Which process am I?
    process_id = int(os.environ.get('SLURM_ARRAY_TASK_ID',1))-1
    # process_id is the int value specifying the index of one particular job
    # this number is pulled from the slurm job array, which is indexed from 1
    # so we have to do minus 1 at the end to be consistent with python index 0
    test_case_id = int(process_id % 20)
    # not perturbing for now because only do it once and 
    # not testing for convergence yet --> to be done
    # p0 = dadi.Misc.perturb_params(p, lower_bound=lb, upper_bound=ub)
    # run optimization for each job and record time
    start = time.time()
    popt, llnlopt = dadi.Inference.opt(p0_list[process_id], 
        fs_list[test_case_id], func_ex, pts_l, lower_bound=lb, upper_bound=ub)
    end = time.time()-start

    print("Test case #:", test_case_id + 1)
    print("True params:", [round(val,5) for val in p_true_list[test_case_id]])
    print("Data Max Log Likelihood:", 
        round(dadi.Inference.ll(fs_list[test_case_id], fs_list[test_case_id]),5))
    print("Starting params:",[round(p,5) for p in p0_list[process_id]])
    print("Optimized params:",[round(p,5) for p in popt])
    print("Model Max Log Likelihood:", round(llnlopt,5))   

    # make lists to save the results
    # results_list = []
    # p1_results_list, p2_results_list, p3_results_list = [], [], []

    # THREE DIFFERENT STARTING POINTS: print out results
    # 1. Generic starting point for regular dadi procedure
    if process_id < 20:
        print('Optimization time for dadi only: {0:.2f}s'.format(end),'\n') 
        # p1_results_list.append(popt)

    # 2. Starting with Prediction from rfr trained on theta=1
    if process_id >= 20 and process_id < 40:
        print('Optimization time for dadi + RFR_1: {0:.2f}s'.format(end),'\n')
        # p2_results_list.append(popt)

    # 3. Average prediction from all four RFR trained on different thetas
    if process_id >= 40:    
        print('Optimization time for dadi + 4_RFR_avg: {0:.2f}s'.format(end),
                                                                        '\n')
        # p3_results_list.append(popt)

    # for i in range(20):
    #   results_list.append((p_true_list[i], p1_results_list[i],
        #            p2_results_list[i], p3_results_list[i]))

    # pickle.dump(results_list, open(
     #   '/groups/rgutenk/lnt/projects/ml-dadi/results/benchmark_out', 'wb'), 2)