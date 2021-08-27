import numpy as np
import matplotlib.pyplot as plt
import ml_models, data_manip
# from scipy import stats

def plot_accuracy_single(true,pred,size= [8,2,20],
                        log=False,r2=None,msle=None,rho=None,c=None):
    '''
    Plot a single true vs. predict panel for one train:test pair
    true, pred = list of true and predicted values for one param,
    which can be obtained from sort_by_param;
    r2: one r2 score for one param of one train:test pair
    msle: one msle score for one param of one train:test pair
    size = [dots_size, line_width, font_size]
    e.g size = [8,2,20] for 4x4
    size= [20,4,40] for 2x2
    '''
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal','box')
    if c is None:
        plt.scatter(true, pred, s=size[0]*2**3) # 's' to change dots size
    else:
        plt.scatter(true, pred, c=c, vmax=5, s=size[0]*2**3) #vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r'$\frac{T}{ν}$', fontweight='bold', fontsize=size[2])
    # axis labels to be customized
    plt.xlabel("true", fontweight='bold')
    plt.ylabel("predicted", fontweight='bold')

    # only plot in log scale if log specified for the param
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(true+pred)*10**-0.5, max(true+pred)*10**0.5])
        plt.ylim([min(true+pred)*10**-0.5, max(true+pred)*10**0.5])
    else:
        # axis scales customized to data
        plt.xlim([min(true+pred)-0.5, max(true+pred)+0.5])
        plt.ylim([min(true+pred)-0.5, max(true+pred)+0.5])
    # plot a slope 1 line
    plt.axline((0,0),(1,1),linewidth=size[1])
    if r2 != None:
        plt.text(0.4, 0.9, "\n\n" + r'$R^{2}$: ' + str(round(r2,4)), horizontalalignment='center', verticalalignment='center', 
        fontsize=size[2], transform = ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "ρ: " + str(round(rho,4)), horizontalalignment='center', verticalalignment='center', 
        fontsize=size[2],transform = ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle,4)), horizontalalignment='center', verticalalignment='center', 
        fontsize=size[2],transform = ax.transAxes)  

def plot_accuracy_multi(list_models, list_test_dict, params, model_name, logs,
                        size=((30, 20), (20, 80), (8,2,20), (24,22)), 
                        r2=True, msle=False, rho=True, c=False):
    '''list_models = list_mlpr or list_rfr
    params = ['s', r'$ν_1$', r'$ν_2$', 'T', 'm12', 'm21']
    model_name = text of title of the plot
    logs = [False, True, True, False, False, False]
    size = ((title_font_size, title_pad), (axis_font_size, axis_pad), single_plot_size_tuple, fig_size)
    '''
    for i in range(len(params)):
        plt.figure(i+1, figsize=size[3], dpi=300)
        plt.gca().set_title(params[i]+' '+model_name, fontsize=size[0][0] , fontweight='bold', pad=size[0][1])
        ax = plt.gca()
        ax.set_xlabel("TEST Variance: increase left to right", 
                fontsize=size[1][0] , fontweight='bold', labelpad=size[1][1])
        ax.set_ylabel("TRAIN Variance: increase bottom to top", 
                fontsize=size[1][0] , fontweight='bold', labelpad=size[1][1])
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xticks([])
        plt.yticks([])
        plt.rcParams.update({'font.size': size[1][0]})
        plt.rcParams.update({'font.weight': 'bold'})

    # testing and plotting
    count_pos = 1
    for model in reversed(list_models):# flip the order of variance for plotting
        for test_dict in list_test_dict:
            # y_true, y_pred = ml_models.model_test(model, test_dict)
            param_true, param_pred = ml_models.model_test(model, test_dict, sort=True)
            # sort test results by param for plotting
            # param_true, param_pred = data_manip.sort_by_param(y_true, y_pred)

            # Colorbar setting is only for 1D_2epoch model currently:        
            if c: # make list of T/nu based on param_true values
                T_over_nu =[T/10**nu for T, nu in zip(param_true[1], param_true[0])]
            else:
                T_over_nu = None
            
            # PLOT MULTIPLE SUBPLOT
            for i in range(len(params)):
                plt.figure(i+1).add_subplot(len(list_models), 
                                            len(list_test_dict), count_pos)
                if logs[i]: # convert log-scale values back to regular scale
                    plot_p_true = [10**p_true for p_true in param_true[i]]
                    plot_p_pred = [10**p_pred for p_pred in param_pred[i]]
                else:
                    plot_p_true = param_true[i]
                    plot_p_pred = param_pred[i]
                if r2:
                    r2_to_plot = ml_models.r2(plot_p_true, plot_p_pred)[0]
                else:
                    r2_to_plot = None
                if msle: # note: msle often cannot be calculated on log-scale 
                    # values because values need to be non-neg
                    msle_to_plot = ml_models.msle(plot_p_true, plot_p_pred)[0]
                else:
                    msle_to_plot = None
                if rho:
                    rho_to_plot = ml_models.rho(plot_p_true, plot_p_pred)
                    # rho_to_plot = stats.spearmanr(plot_p_true, plot_p_pred)[0]
                else:
                    rho_to_plot = None
                plot_accuracy_single(plot_p_true, plot_p_pred, size[2],
                                        log=logs[i], r2=r2_to_plot,
                                        msle=msle_to_plot, rho=rho_to_plot, 
                                        c=T_over_nu)
            count_pos += 1

# plotting for bootstrap, to be cleaned up
def plot_intervals(int_arr_all, params, size=50):
    '''
    Plot all of the [size] intervals for the specified theta
    int_arr_all: all_intervals_by_param output from bootstrap_intervals()
    params: list of params used in the model as strings, 
    e.g, ['nu1', 'nu2', 'T', 'm']
    size: number of intervals to plot; take the first [size] results instead of using all 200
    (or however many bootstrap samples were generated)
    '''
    x = range(size)
    for param,int_arr in zip(params, int_arr_all):
        int_arr = np.array(int_arr[:size])
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
        plt.show()            

def plot_distribution(bootstrap_pred_i, params, n):
    '''
    Plots the distribution of all of the bootstraps for some specified sample n 
    bootstrap_pred_i: a single dict for 1 theta case 
    from the list of dictionaries output from bootstrap_predictions()
    params: list of params used in the model as strings, 
    e.g, ['nu1', 'nu2', 'T', 'm']
    theta_i: the index of desired theta to use from theta_list
    n: the index to use from the bootstrap results. 
    For example, if bootstrap_data generated a total of 100 original fs 
    and 300 bootstrap fs for each of the originals, a valid n would be 0-99,
    and the distribution of all 300 predictions for all parameters for that
    specific fs will be plotted.
    '''
    p_orig = list(bootstrap_pred_i.keys())[n]
    dist = bootstrap_pred_i[p_orig][1]
    dist = np.array(dist)
    dist = dist.transpose(1, 0)
    fig, axs = plt.subplots(len(params), figsize=(5, 8))
    for i in range(len(params)):
        true = p_orig[i]
        axs[i].axvline(x=true, c='red', label='true')
        orig = bootstrap_pred_i[p_orig][0][i]
        axs[i].axvline(x=orig, c='blue', label='original')
        axs[i].hist(dist[i], bins='sqrt')
        axs[i].set_title(params[i])
    handles, labels = axs[len(params)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'results/{datafile}_distribution_{n}_theta{theta_list[theta_i]}.png')

def plot_coverage(bootstrap_pred_i, params, expected):
    '''
    Plots coverage results for all the parameters in the model
    bootstrap_pred_i: a single dict for 1 theta case 
    from the list of dictionaries output from bootstrap_predictions()
    params: list of params used in the model as strings, e.g, ['nu1', 'nu2', 'T', 'm']
    expected: list of the expected coveragte percentages to look at e.g., [30, 50, 80, 95]
    '''
    observed = [[] for x in range(len(params))]
    for perc in expected:
        ints = data_manip.bootstrap_intervals(bootstrap_pred_i, 
                                                params, percentile=perc)
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
