import numpy as np
import dadi
import pickle
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

theta_list = [100, 1000, 10000] # global variable
# Note: needs to match the list used in *bootstrap_data.py

save_to = '../../results/bootstrapping/' # global variable
# path to directory where you want all of the results to be saved

def bootstrap_predictions(ml_file, bs_data_file):
    '''
    Runs the ML predictions for all of the generated bootstrap samples and saves
    to the results folder.
    ml_file: path to the trained ML model
    bs_data_file: path to the bootstrap data.
    '''
    ml_model = pickle.load(open(ml_file, 'rb'))
    bs_samples = pickle.load(open(bs_data_file, 'rb'))
    # The form of bs_samples is a list of dictionaries in the order theta = [100, 1000, 10000]
    # and each dictionary is of the form {(true param) : [original fs, [bootstrap fs x 200]] x 200}
    # unless otherwise specified in *bootstrap_data.py
    # NOTE: none of the fs in bs_samples are normed
    pred_dict_list = []
    for d in bs_samples:
        pred_dict = {}
        for true_p, all_fs in d.items():
            orig_fs = all_fs[0]   # single fs
            orig_fs = orig_fs/orig_fs.sum()
            bs_fs_list = all_fs[1]          # list length 200
            orig_pred = ml_model.predict([orig_fs.flatten()]).flatten()
            bs_pred_list = []
            for bs_fs in bs_fs_list:
                bs_fs = bs_fs/bs_fs.sum()
                bs_pred = ml_model.predict([bs_fs.flatten()]).flatten()
                bs_pred_list.append(bs_pred)
            pred_dict[true_p] = [orig_pred, bs_pred_list]
        pred_dict_list.append(pred_dict)
    file_name = ml_file[ml_file.rindex('/')+1:] # should get name after last slash
    pickle.dump(pred_dict_list, open(f'{save_to}bootstrap_{file_name}', 'wb'), 2)

def bootstrap_intervals(datafile, params, theta, percentile=95):
    '''
    Generates a list of intervals sorted by param containing true value, original value, 
    lower bound, and upper bound
    datafile: path to the bootstrap prediction results (obtained from bootstrap_predictions())
    params: list of params used in the model as strings, e.g, ['nu1', 'nu2', 'T', 'm']
    theta: the desired theta to use (must be from theta_list above)
    percentile: the desired % interval 
    '''
    bs_results = pickle.load(open(datafile, 'rb'))
    pred_theta = bs_results[theta_list.index(theta)] # dict of preds for fs scaled by theta with index [100, 1000, 10000]
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
    # list length 4, inner list length 200, inner list length 4
    # e.g., outermost list: [nu1, nu2, T, m]
    # where nu1 = [list 1, list 2 ...] where list 1 = [true, orig, lower, upper]

def plot_intervals(datafile, params, theta, size=50):
    '''
    Plot all of the [size] intervals for the specified theta
    datafile: path to the bootstrap prediction results (obtained from bootstrap_predictions())
    params: list of params used in the model as strings, e.g, ['nu1', 'nu2', 'T', 'm']
    theta: the desired theta to use (must be from theta_list above)
    size: number of intervals to plot; take the first [size] results instead of using all 200
    (or however many bootstrap samples were generated)
    '''
    int_arr_all = bootstrap_intervals(datafile, params, theta)
    int_arr_all = np.array(int_arr_all)
    x = range(size)
    for param,int_arr in zip(params, int_arr_all):
        int_arr = int_arr[:size]
        int_arr = int_arr.transpose(1, 0)
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
        file_name = datafile[datafile.rindex('/')+1:] # should get the name after last slash
        plt.savefig(f'{save_to}{file_name}_{param}_{size}intervals_theta{theta}.png')

def plot_distribution(datafile, params, theta, n):
    '''
    Plots the distribution of all of the bootstraps for some specified sample n 
    datafile: path to the bootstrap prediction results (obtained from bootstrap_predictions())
    params: list of params used in the model as strings, e.g, ['nu1', 'nu2', 'T', 'm']
    theta: the desired theta to use (must be from theta_list above)
    n: the index to use from the bootstrap results. For example, if bootstrap_data generated
    a total of 100 original fs and 300 bootstrap fs for each of the originals, a valid n would
    be 0-99, and the distribution of all 300 predictions for all parameters for that specific fs
    will be plotted.
    '''
    bs_results = pickle.load(open(datafile, 'rb'))
    pred_theta = bs_results[theta_list.index(theta)]
    p_orig = list(pred_theta.keys())[n]
    dist = pred_theta[p_orig][1]
    dist = np.array(dist)
    dist = dist.transpose(1, 0)
    fig, axs = plt.subplots(4, figsize=(5, 8))
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
    file_name = datafile[datafile.rindex('/')+1:] # should get the name after last slash
    plt.savefig(f'{save_to}{file_name}_distribution{n}_theta{theta}.png')

def sort_by_param(p_sets):
    '''
    util used by bootstrap_intervals
    '''
    p_sorted = [] # list by param
    for i in range(len(p_sets[0])):
        p = [] # single param list (e.g., nu1)
        for p_set in p_sets:
            p.append(p_set[i])
        p_sorted.append(p)
    return p_sorted

def plot_coverage(datafile, params, theta, desired):
    '''
    Plots coverage results for all the parameters in the model
    datafile: path to the bootstrap prediction results (obtained from bootstrap_predictions())
    params: list of params used in the model as strings, e.g, ['nu1', 'nu2', 'T', 'm']
    theta: the desired theta to use (must be from theta_list above)
    desired: list of the desired coveragte percentages to look at e.g., [30, 50, 80, 95]
    '''
    observed = [[] for x in range(len(params))]
    for perc in desired:
        ints = bootstrap_intervals(datafile, params, theta, percentile=perc)
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
    ax.plot(desired, desired, label='match', color='gray')
    for i in range(len(params)):
        ax.plot(desired, observed[i], label=params[i], marker='o', color=colors[i])
    ax.legend()
    ax.set_xlabel("desired")
    ax.set_ylabel("observed")
    plt.xticks(np.arange(min(desired), max(desired)+1, 5))
    plt.yticks(np.arange(min(desired), max(desired)+1, 5))
    plt.tight_layout()
    file_name = datafile[datafile.rindex('/')+1:] # should get the name after last slash
    plt.savefig(f'{save_to}{file_name}_coverage_theta{theta}.png')

if __name__ == '__main__': 
    # generate file containing bootstrap prediction results 
    ml_file = '../../data/nn_1000'
    bs_data_file = '../../data/test_set_bootstraps'
    bootstrap_predictions(ml_file, bs_data_file)

    # use the different plotting functions
    datafile = f'{save_to}bootstrap_nn_1000'
    params = ['nu1', 'nu2', 'T', 'm']
    theta = 1000
    # plot first 100 intervals for theta = 1000
    plot_intervals(datafile, params, theta, size=100)
    # plot distribution for theta = 1000, chosen sample index = 23
    plot_distribution(datafile, params, theta, 23)
    # plot coverage for theta = 1000
    desired = [95, 90, 80, 50, 30, 15]
    plot_coverage(datafile, params, theta, desired)

    # if you would like to generate plots for all thetas at once, use a loop
    for theta in theta_list:
        plot_coverage(datafile, params, theta, desired)
        plot_distribution(datafile, params, theta, 0) # 0th index
        plot_intervals(datafile, params, theta) # size 50