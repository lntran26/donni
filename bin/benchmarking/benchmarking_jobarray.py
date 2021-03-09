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
    test_set = pickle.load(open(
        '/groups/rgutenk/lnt/projects/ml-dadi/data/benchmarking_test_set','rb'))
    p_true_list = [test_set[i][0] for i in range(20)]
    fs_list = [test_set[i][1] for i in range(20)]
    p0_list = [test_set[i][2] for i in range(60)]

    # benchmarking
    # common process: set up dadi
    # designate demographic model, sample size, bounds, extrapolation grid 
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    func_ex = dadi.Numerics.make_extrap_func(func)
    lb, ub = [1e-2,1e-2,0.1,1], [1e2,1e2,2,10]

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

    # THREE DIFFERENT STARTING POINTS: print out results
    # 1. Generic starting point for regular dadi procedure
    if process_id < 20:
        print('Optimization time for dadi only: {0:.2f}s'.format(end),'\n') 

    # 2. Starting with Prediction from rfr trained on theta=1
    if process_id >= 20 and process_id < 40:
        print('Optimization time for dadi + RFR_1: {0:.2f}s'.format(end),'\n')

    # 3. Average prediction from all four RFR trained on different thetas
    if process_id >= 40:    
        print('Optimization time for dadi + 4_RFR_avg: {0:.2f}s'.format(end),
                                                                        '\n')