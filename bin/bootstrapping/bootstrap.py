import numpy as np
import dadi
import pickle
import matplotlib.pyplot as plt

def bootstrap_predictions(trained_model, bootstrap_data):
    '''
    trained_model is a single trained ML such as rfr_1000
    bootstrap_data is a list of dictionaries of the format 
    true_p:[orig_fs,[200 bootstrapped fs from orig_fs]]
    for each theta case in the list.
    return: a list of bootstrap_pred dictionaries of the same
    format as bootstrap_data
    '''
    bootstrap_pred = []
    for theta_case in bootstrap_data:
        pred_dict_bstr = {}
        for true_p, all_fs in theta_case.items():
            # load elements from bootstrap data
            orig_fs = all_fs[0] # single original fs
            list_bstr_fs = all_fs[1] # list of 200 bootstrapped fs from orig_fs
            # run prediction for original fs
            orig_fs = orig_fs/orig_fs.sum() # normalize for prediction
            orig_fs_pred = trained_model.predict([orig_fs.flatten()]).flatten()
            # run prediction for bootstrapped fs
            list_bstr_fs_pred = []
            for bstr_fs in list_bstr_fs:
                bstr_fs = bstr_fs/bstr_fs.sum() # normalize for prediction
                bstr_fs_pred = trained_model.predict([bstr_fs.flatten()]).flatten()
                list_bstr_fs_pred.append(bstr_fs_pred)
            # save all predictions into a dictionary for each theta case
            pred_dict_bstr[true_p] = [orig_fs_pred, list_bstr_fs_pred]
        # save all dictionaries into a list for multiple theta cases
        bootstrap_pred.append(pred_dict_bstr)
    return bootstrap_pred

def sort_by_param_bstr(p_sets):
    p_sorted = [] # list by param
    for i in range(len(p_sets[0])):
        p = [] # single param list (e.g., nu1)
        for p_set in p_sets:
            p.append(p_set[i])
        p_sorted.append(p)
    return p_sorted

def bootstrap_intervals(datafile, params, theta_i, percentile=95):
    '''
    Input: bootstrap prediction results
    datafile: filename of bootstrap prediction result
    params: list of params, e.g., [nu1, nu2, T, m]
    theta_i: integer 0, 1, 2 ,3... to iterate through theta cases

    Output: all_intervals_by_param: list of length 4 for params: nu1, nu2, T, m;
    where nu1 = [list 1, list 2 ..., list 200] 
    where list 1 = [true, orig, lower, upper]
    '''
    bs_results = pickle.load(open(datafile, 'rb'))
    pred_theta = bs_results[theta_i] # dict of preds for fs scaled by theta with index [100, 1000, 10000]
    keys = list(pred_theta.keys())
    # all_intervals_by_param = [[], [], [], []]
    all_intervals_by_param = []
    for p in range(len(params)):
        all_intervals_by_param.append([])
    for j,key in enumerate(keys):
        preds = pred_theta[key] # list of predictions from [orig fs pred, [200 boot fs pred]]
        #print(f"true: {key}")
        #print(f"orig: {preds[0]}")
        bs_by_param = sort_by_param_bstr(preds[1])
        bounds = (100 - percentile) / 2 / 100 
        offset = int(len(bs_by_param[0]) * bounds) - 1 # e.g., 5th value is index 4
        for i,cur_p in enumerate(params):
            all_cur_p = bs_by_param[i] # sorted by current param, e.g., all nu1 preds sorted
            all_cur_p.sort()
            true = key[i]
            orig = preds[0][i]
            low = all_cur_p[offset]
            high = all_cur_p[len(all_cur_p) - offset]
            #print(f'---------{cur_p}---------')
            #print(f'true: {true:.4f}')
            #print(f'orig: {orig:.4f}')
            #print(f'{percentile}% CI: {low:.4f}-{high:.4f}')
            interval = [true, orig, low, high]
            all_intervals_by_param[i].append(interval)
    return all_intervals_by_param

def plot_distribution(datafile, params, theta_i, n):
    bs_results = pickle.load(open(datafile, 'rb'))
    pred_theta = bs_results[theta_i]
    p_orig = list(pred_theta.keys())[n]
    dist = pred_theta[p_orig][1]
    dist = np.array(dist)
    dist = dist.transpose(1, 0)
    fig, axs = plt.subplots(len(params), figsize=(5, 8))
    for i in range(len(params)):
        true = p_orig[i]
        axs[i].axvline(x=true, c='red', label='true')
        orig = pred_theta[p_orig][0][i]
        axs[i].axvline(x=orig, c='blue', label='original')
        axs[i].hist(dist[i], bins='sqrt')
        axs[i].set_title(params[i])
    handles, labels = axs[len(params)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'results/{datafile}_distribution_{n}_theta{theta_list[theta_i]}.png')

def plot_coverage(datafile, params, expected, theta_i):
    observed = [[] for x in range(len(params))]
    for perc in expected:
        ints = bootstrap_intervals(datafile, params, theta_i, percentile=perc)
        size = len(ints[0])
        for p_i,int_arr,param in zip(range(len(params)),ints,params): # list by params
            covered = 0
            int_arr = np.array(int_arr)
            int_arr = int_arr.transpose(1, 0)
            # now in the form [all true], [all orig], [all lower], [all upper]; indices match up
            for i in range(len(int_arr[0])):
                if int_arr[0][i] >= int_arr[2][i] and int_arr[0][i] <= int_arr[3][i]:
                    covered += 1
            observed[p_i].append(covered/2)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ['red', 'blue', 'green', 'yellow']
    ax.plot(expected, expected, label='match', color='gray')
    for i in range(len(params)):
        ax.plot(expected, observed[i], label=params[i], marker='o', color=colors[i])
    ax.legend()
    ax.set_xlabel("expected")
    ax.set_ylabel("observed")
    plt.xticks(np.arange(min(expected), max(expected)+1, 5))
    plt.yticks(np.arange(min(expected), max(expected)+1, 5))
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'results/{datafile}_coverage_theta{theta_list[theta_i]}.png')