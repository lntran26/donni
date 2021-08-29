import numpy as np
import matplotlib.pyplot as plt
import ml_models
from ml_models import model_test


def plot_accuracy_single(x, y, size=[8, 2, 20], x_label="true",
                         y_label="predict", log=False,
                         r2=None, msle=None, rho=None, c=None):
    '''
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    msle: msle score for x and y (x, y need to be non-log, i.e. non-neg)
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    '''

    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal', 'box')

    # plot data points in a scatter plot
    if c is None:
        plt.scatter(x, y, s=size[0]*2**3)  # 's' specifies dots size
    else:  # condition to add color bar
        plt.scatter(x, y, c=c, vmax=5, s=size[0]*2**3)  # vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r'$\frac{T}{ν}$',
                          fontweight='bold', fontsize=size[2])

    # axis label texts
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, fontweight='bold')

    # only plot in log scale if log specified for the param
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.ylim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
    else:
        # axis scales customized to data
        plt.xlim([min(x+y)-0.5, max(x+y)+0.5])
        plt.ylim([min(x+y)-0.5, max(x+y)+0.5])

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (1, 1), linewidth=size[1])

    # plot scores if specified
    if r2 != None:
        plt.text(0.4, 0.9, "\n\n" + r'$R^{2}$: ' + str(round(r2, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "ρ: " + str(round(rho, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)


def plot_accuracy_multi_data(data, logs, params,
                             size=((30, 20), (20, 80), (8, 2, 20), (24, 22)),
                             title=None, x_label=None, y_label=None,
                             sub_x_label="true", sub_y_label="predict",
                             r2=False, msle=False, rho=True, c=False):
    '''
    Plot multiple panels of x vs y scatter plots with correlation scores 
    from raw input data as described below.

    data: list of lists to dictate the number and order of subplots 
        in the rows and columns of the main plot, 
        with the outer list length = # rows and inner list length = # columns. 
        Each element of the inner list is a tuple of size 2, e.g. (true, pred),
        containing the data points sorted by params to be plotted, 
        e.g. true = ([100_true_nu_values], [100_true_T_values])
    logs: a list to specify which param is in log scale, e.g. [True, False] 
    params: a list of param names, e.g. [r'$ν_1$', 'T']
    size = ((title_font_size, title_pad), (axis_font_size, axis_pad),
        single_plot_size_tuple, fig_size)
    title: text of title on main figure, e.g. "MLPR adam"
    x_label: text of x axis label on main figure, e.g. "Test variance"
    y_label: text of y axis label on main figure, e.g. "Train variance"
    '''

    for i, param in enumerate(params):  # plot one figure per param in list
        plt.figure(i+1, figsize=size[3], dpi=300)

        # set plot title
        plt.gca().set_title(
            params[i]+' '+title, fontsize=size[0][0],
            fontweight='bold', pad=size[0][1])

        # set plot axes
        ax = plt.gca()
        # set axis labels
        ax.set_xlabel(x_label, fontsize=size[1][0],
                      fontweight='bold', labelpad=size[1][1])
        ax.set_ylabel(y_label, fontsize=size[1][0],
                      fontweight='bold', labelpad=size[1][1])
        # set axis values
        plt.rcParams.update({'font.size': size[1][0]})
        plt.rcParams.update({'font.weight': 'bold'})

        # make ticks and tick labels invisible
        for key, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.xticks([])
        plt.yticks([])

    # Populate main figure with subplots using double for loops
    count_pos = 1
    for row in data:
        for column in row:
            # read data from input
            param_true, param_pred = column

            # Colorbar setting is only for 1D_2epoch & 1D_growth models
            if c:  # make list of T/nu based on param_true values
                T_over_nu = [T/10**nu for T,
                             nu in zip(param_true[1], param_true[0])]
            else:
                T_over_nu = None

            # Plot multiple subplots
            for i, param in enumerate(params):
                plt.figure(i+1).add_subplot(len(data), len(row), count_pos)

                # handling log-scale data
                if logs[i]:  # convert log-scale values back to regular scale
                    plot_p_true = [10**p_true for p_true in param_true[i]]
                    plot_p_pred = [10**p_pred for p_pred in param_pred[i]]
                else:  # leave as is if values not in log scale
                    plot_p_true = param_true[i]
                    plot_p_pred = param_pred[i]

                # handling scores
                if r2:
                    r2_to_plot = ml_models.r2(plot_p_true, plot_p_pred)[0]
                else:
                    r2_to_plot = None
                if msle:  # msle can only be calculated on non-neg values
                    msle_to_plot = ml_models.msle(plot_p_true, plot_p_pred)[0]
                else:
                    msle_to_plot = None
                if rho:
                    rho_to_plot = ml_models.rho(plot_p_true, plot_p_pred)
                else:
                    rho_to_plot = None

                # plot a single subplot
                plot_accuracy_single(plot_p_true, plot_p_pred, size[2],
                                     x_label=sub_x_label, y_label=sub_y_label,
                                     log=logs[i], r2=r2_to_plot,
                                     msle=msle_to_plot, rho=rho_to_plot,
                                     c=T_over_nu)
            count_pos += 1


def plot_accuracy_multi(list_models, list_test_dict, logs, params,
                        size=((30, 20), (20, 80), (8, 2, 20), (24, 22)),
                        title=None, x_label=None, y_label=None,
                        r2=False, msle=False, rho=True, c=False):
    '''
    Plot multiple panels of true vs pred plots with correlation scores 
    from specified trained ML models and test data sets.

    list_models: list of trained ML models to be tested, 
        e.g. list_mlpr or list_rfr
    list_test_dict: list of test data sets, each set is a dictionary of
        true values as keys and SFS as test data
    Other arguments are the same as in plot_accuracy_multi_data
    '''

    # Generate test prediction results to be plotted
    data = []
    # Use reversed() to flip the order of train variance for plotting
    for model in reversed(list_models):
        test_res = []
        for test_dict in list_test_dict:
            # pass test data into trained ML model to make predictions
            # sort=True to sort resulting data by params
            test_res.append(model_test(model, test_dict, sort=True))
        data.append(test_res)

    # Plot the results
    plot_accuracy_multi_data(data, logs, params, size, title,
                             x_label=x_label, y_label=y_label, 
                             r2=r2, msle=msle, rho=rho, c=c)


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
    for param, int_arr in zip(params, int_arr_all):
        int_arr = np.array(int_arr[:size])
        int_arr = int_arr.transpose(1, 0)
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
        minor_ticks = np.arange(0, size)
        major_ticks = np.arange(0, size, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=x)
        ax.grid(which='both')

        ax.scatter(x, int_arr[0], c="red", label="x")  # x
        neg_int = int_arr[1] - int_arr[2]
        pos_int = int_arr[3] - int_arr[1]
        ax.errorbar(x, int_arr[1], yerr=[
                    neg_int, pos_int], fmt='bo', label="orig")
        ax.set_title(param)
        ax.legend()
        plt.show()


def plot_distribution(bootstrap_y_i, params, n):
    '''
    Plots the distribution of all of the bootstraps for some specified sample n 
    bootstrap_y_i: a single dict for 1 theta case 
    from the list of dictionaries output from bootstrap_yictions()
    params: list of params used in the model as strings, 
    e.g, ['nu1', 'nu2', 'T', 'm']
    theta_i: the index of desired theta to use from theta_list
    n: the index to use from the bootstrap results. 
    For example, if bootstrap_data generated a total of 100 original fs 
    and 300 bootstrap fs for each of the originals, a valid n would be 0-99,
    and the distribution of all 300 yictions for all parameters for that
    specific fs will be plotted.
    '''
    p_orig = list(bootstrap_y_i.keys())[n]
    dist = bootstrap_y_i[p_orig][1]
    dist = np.array(dist)
    dist = dist.transpose(1, 0)
    fig, axs = plt.subplots(len(params), figsize=(5, 8))
    for i in range(len(params)):
        x = p_orig[i]
        axs[i].axvline(x=x, c='red', label='x')
        orig = bootstrap_y_i[p_orig][0][i]
        axs[i].axvline(x=orig, c='blue', label='original')
        axs[i].hist(dist[i], bins='sqrt')
        axs[i].set_title(params[i])
    handles, labels = axs[len(params)-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'results/{datafile}_distribution_{n}_theta{theta_list[theta_i]}.png')


def plot_coverage(bootstrap_y_i, params, expected):
    '''
    Plots coverage results for all the parameters in the model
    bootstrap_y_i: a single dict for 1 theta case 
    from the list of dictionaries output from bootstrap_yictions()
    params: list of params used in the model as strings, e.g, ['nu1', 'nu2', 'T', 'm']
    expected: list of the expected coveragte percentages to look at e.g., [30, 50, 80, 95]
    '''
    observed = [[] for x in range(len(params))]
    for perc in expected:
        ints = data_manip.bootstrap_intervals(bootstrap_y_i,
                                              params, percentile=perc)
        size = len(ints[0])
        # list by params
        for p_i, int_arr, param in zip(range(len(params)), ints, params):
            covered = 0
            int_arr = np.array(int_arr)
            int_arr = int_arr.transpose(1, 0)
            # now in the form [all x], [all orig], [all lower], [all upper]; indices match up
            for i in range(len(int_arr[0])):
                if int_arr[0][i] >= int_arr[2][i] and int_arr[0][i] <= int_arr[3][i]:
                    covered += 1
            observed[p_i].append(covered/2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['red', 'blue', 'green', 'yellow']
    ax.plot(expected, expected, label='match', color='gray')
    for i in range(len(params)):
        ax.plot(expected, observed[i], label=params[i],
                marker='o', color=colors[i])
    ax.legend()
    ax.set_xlabel("expected")
    ax.set_ylabel("observed")
    plt.xticks(np.arange(min(expected), max(expected)+1, 5))
    plt.yticks(np.arange(min(expected), max(expected)+1, 5))
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'results/{datafile}_coverage_theta{theta_list[theta_i]}.png')
