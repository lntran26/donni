import util
import dadi
import pickle
import random
from copy import deepcopy
import numpy as np

if __name__ == '__main__': 
    # import test sets previously generated for 2D-split-migration
    list_test_dict = pickle.load(open(
        '../../data/2d-splitmig/test-data-corrected-2','rb'))
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
                '../../data/2d-splitmig/list-rfr','rb'))
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
    fs_list_ext = fs_list * 3
    p_true_list_ext = p_true_list * 3

    # make test set:
    test_set = []
    for i in range(60):
        test_set.append((p_true_list_ext[i], fs_list_ext[i], p0_list[i]))
    pickle.dump(test_set, open('benchmarking_test_set_3', 'wb'), 2)