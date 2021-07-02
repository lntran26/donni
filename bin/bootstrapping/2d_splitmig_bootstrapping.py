import numpy as np
import dadi
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def bootstrap_predictions():
    ml_name = 'rf' # 'nn'
    ml = pickle.load(open('/groups/rgutenk/lnt/projects/ml-dadi/data/list-rfr', 'rb'))
    ml = ml[0] # for list-rfr, use trained on theta 1
    # maybe also run a test with nn trained on 10,000
    bs_samples = pickle.load(open('../../data/test_set_bootstraps', 'rb'))
    # form of bs_samples is list of dictionaries in the order 100, 1000, 10000
    # each dictionary is of the form {(true param) : [original fs, [bootstrap fs x 200]] x 200}
    
    # TODO: run the original synthetic data set through the NN and save the prediction
    # then, run the 200 bootstrap fs through the NN, sort the predictions by each of the 
    # 4 params, and save those (create the 95% confidence interval by choosing the 5th
    # and 195th as bounds)

    # same form as bs_samples except do [original pred params, [bootstrap pred params x 200]]

    # NOTE: none of the fs in bs_samples are normed

    thetas = [100, 1000, 10000]
    pred_dict_list = []
    for d in bs_samples:
        pred_dict = {}
        for true_p, all_fs in d.items():
            orig_fs = all_fs[0]   # single fs
            orig_fs = orig_fs/orig_fs.sum()
            bs_fs_list = all_fs[1]          # list length 200
            orig_pred = ml.predict([orig_fs.flatten()]).flatten()
            bs_pred_list = []
            for bs_fs in bs_fs_list:
                bs_fs = bs_fs/bs_fs.sum()
                bs_pred = ml.predict([bs_fs.flatten()]).flatten()
                bs_pred_list.append(bs_pred)
            pred_dict[true_p] = [orig_pred, bs_pred_list]
        pred_dict_list.append(pred_dict)
    
    pickle.dump(pred_dict_list, open(f'../../results/bootstrap_{ml_name}', 'wb'), 2)

def bootstrap_intervals(params, percentile=95):
    '''
    pass a list of params as argument, e.g., [nu1, nu2, T, m]
    '''
    bs_results = pickle.load(open('bootstrap', 'rb'))
    #pred_theta = bs_results[1] # dict of preds for fs scaled by theta 1000
    pred_theta = bs_results[2] # dict of preds for fs scaled by theta 10000
    #pred_theta = bs_results[0] # dict of preds for fs scaled by theta 100
    keys = list(pred_theta.keys())
    all_intervals_by_param = [[], [], [], []]
    for j,key in enumerate(keys):
        preds = pred_theta[key] # list of predictions from [orig fs pred, [200 boot fs pred]]
        #print(f"true: {key}")
        #print(f"orig: {preds[0]}")
        bs_by_param = sort_by_param(preds[1])
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
    # list length 4, inner list length 200, inner list length 4 [nu1, nu2, T, m]
    # where nu1 = [list 1, list 2 ...] where list 1 = [true, orig, lower, upper]

def plot_distribution(params, n):
    bs_results = pickle.load(open('bootstrap', 'rb'))
    #pred_100 = bs_results[0]
    pred_1000 = bs_results[1]
    #pred_10000 = bs_results[2]
    p_orig = list(pred_1000.keys())[n]
    dist = pred_1000[p_orig][1]
    dist = np.array(dist)
    dist = dist.transpose(1, 0)
    fig, axs = plt.subplots(4)
    for i in range(len(params)):
        true = p_orig[i]
        axs[i].axvline(x=true, c='red', label='true')
        orig = pred_1000[p_orig][0][i]
        axs[i].axvline(x=orig, c='blue', label='original')
        axs[i].hist(dist[i], bins='sqrt')
        axs[i].set_title(params[i])
    plt.legend()
    plt.tight_layout()
    plt.show()

    

def sort_by_param(p_sets):
    p_sorted = [] # list by param
    for i in range(len(p_sets[0])):
        p = [] # single param list (e.g., nu1)
        for p_set in p_sets:
            p.append(p_set[i])
        p_sorted.append(p)
    return p_sorted

def plot_coverage(params, expected):
    observed = [[] for x in range(len(params))]
    for perc in expected:
        ints = bootstrap_intervals(params, percentile=perc)
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
    ax.plot(expected, expected, label='expected', color='gray')
    for i in range(len(params)):
        ax.plot(expected, observed[i], label=params[i], marker='o', color=colors[i])
    ax.legend()
    plt.xticks(np.arange(min(expected), max(expected)+1, 5))
    plt.yticks(np.arange(min(expected), max(expected)+1, 5))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
    bootstrap_predictions()
    #params = ['nu1', 'nu2', 'T', 'm']
    #expected = [95, 90, 80, 50, 30, 15]
    #plot_coverage(params, expected)
    #plot_distribution(params, 0)

    '''
        # plot all intervals
        x = range(size)
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1,1,1)
        minor_ticks = np.arange(0, size)
        major_ticks = np.arange(0, size, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='both')

        ax.scatter(x, int_arr[0], c="red", label="true") # true
        
        neg_int = int_arr[1] - int_arr[2]
        pos_int = int_arr[3] - int_arr[1]
        ax.errorbar(x, int_arr[1], yerr=[neg_int, pos_int], fmt='bo', label="orig")
        ax.set_title(param)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        '''
            