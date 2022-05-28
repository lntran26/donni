"""Module for using trained MLPR to plot many demographic param predictions"""
import numpy as np
from mapie.metrics import regression_coverage_score
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt


def get_r2(y_true, y_pred):
    """Method to get R2 score for prediction"""

    score = r2_score(y_true, y_pred)
    score_by_param = r2_score(y_true, y_pred, multioutput='raw_values')
    return score, score_by_param


def get_rho(y_true, y_pred):
    """stats.spearmanr returns two values: correlation and p-value
    Here we only want the correlation value"""

    return stats.spearmanr(y_true, y_pred)[0]


def sort_by_param(y_true, y_pred):
    '''
    Sort the predictions by multioutput sklearn mlpr into lists of
    true vs predict values by each param used in the model.
    Returns: param_true and param_pred are each a list of lists,
    each sublist contains true or pred values for one param
    '''
    param_true, param_pred = [], []
    n = 0
    while n < len(y_true[0]):
        param_list_true, param_list_pred = [], []
        for true, pred in zip(y_true, y_pred):
            param_list_true.append(true[n])
            param_list_pred.append(pred[n])
        param_true.append(param_list_true)
        param_pred.append(param_list_pred)
        n += 1
    return param_true, param_pred


def plot_accuracy_single(x, y, size=[8, 2, 20], x_label="true",
                         y_label="predict", log=False,
                         r2=None, msle=None, rho=None, c=None, title=None):
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
        if max(x+y) > 1:
            plt.xlim([min(x+y)-0.5, max(x+y)+0.5])
            plt.ylim([min(x+y)-0.5, max(x+y)+0.5])
        else:
            plt.xlim([min(x+y)-0.05, max(x+y)+0.05])
            plt.ylim([min(x+y)-0.05, max(x+y)+0.05])

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (1, 1), linewidth=size[1])

    # plot scores if specified
    if r2 is not None:
        plt.text(0.4, 0.9, "\n\n" + r'$R^{2}$: ' + str(round(r2, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if rho is not None:
        plt.text(0.4, 0.9, "ρ: " + str(round(rho, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if msle is not None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if title is not None:
        ax.set_title(title, fontsize=size[2], fontweight='bold')


def plot_coverage(cov_scores, alpha, results_prefix, theta=None, params=None):
    """Helper method to plot coverage plot"""

    expected = [100*(1 - a) for a in alpha]
    observed = []
    for cov_score in cov_scores:
        observed.append([s*100 for s in cov_score])

    plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    font = {'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    title = 'coverage'
    if theta:
        title += f', theta {theta}'
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel("expected", fontsize=12, fontweight='bold')
    ax.set_ylabel("observed", fontsize=12, fontweight='bold')

    for i in range(len(cov_scores)):
        label = params[i] if params is not None else 'param '+str(i+1)
        ax.plot(expected, observed[i],
                label=label, marker='o', linewidth=2)
    ax.plot(expected, expected, label='match', linewidth=2, color="black")

    plt.xticks(np.arange(min(expected), max(expected)+5, 10))
    plt.yticks(np.arange(min(expected), max(expected)+5, 10))
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    ax.legend(loc='upper left', fontsize=8)

    plt.savefig(f'{results_prefix}_coverage')
    plt.clf()


def plot(models: list, X_test, y_test, results_prefix, logs, mapie=True,
         coverage=False, theta=None, params=None):
    """Main method to plot both accuracy and coverage plots"""

    # # unpack test_data dict
    # X_test = [np.array(fs).flatten() for fs in test_data.values()]
    # y_test = list(test_data.keys())

    # # parse labels into single list for each param (required for mapie)
    # y_test_unpack = list(zip(*y_test)) if mapie else [y_test]

    # some set alpha
    alpha = [.05, .1, .2, .5, .7, .85]

    # make prediction with trained mlpr models
    c = None
    if mapie:
        all_coverage = []
        c = None
        if len(logs) == 2:  # assumption for now
            T_true = y_test[1]
            nu_true = y_test[0]
            nu_true_delog = [10**nu for nu in nu_true]
            c = [T/nu for T, nu in zip(T_true, nu_true_delog)]
        for model_i, model in enumerate(models):
            true = y_test[model_i]
            if coverage:
                pred, pis = model.predict(X_test, alpha=alpha)
                coverage_scores = [
                    regression_coverage_score(true, pis[:, 0, i], pis[:, 1, i])
                    for i, _ in enumerate(alpha)
                ]
                all_coverage.append(coverage_scores)
            else:
                pred = model.predict(X_test)
            # get scores for normal pred versions (logged)
            r2 = get_r2(true, pred)[0]
            rho = get_rho(true, pred)
            if params is None:
                title = f'param {model_i + 1}'
            else:
                title = f'{params[model_i]}'
            if theta:
                title += f' theta {theta}'
            if model_i < len(logs) - 1 and logs[model_i]:
                # for log params, exponentiate each of the test and pred values
                true_delog = [10**p_true for p_true in true]
                pred_delog = [10**p_pred for p_pred in pred]
                plot_accuracy_single(true_delog, pred_delog, size=[6, 2, 20],
                                     log=True, r2=r2, rho=rho,
                                     title=title, c=c)
            else:
                plot_accuracy_single(list(true), list(pred),
                                     size=[6, 2, 20], log=False,
                                     r2=r2, rho=rho, title=title, c=c)
            plt.savefig(f'{results_prefix}_param_{model_i + 1:02d}_accuracy')
            plt.clf()

        if coverage:
            # plot coverage
            plot_coverage(all_coverage, alpha, results_prefix, theta, params)

    else:  # implement sklearn version
        # for sklearn multioutput, models is a list of one mlpr
        model = models[0]
        # and true is a list of one list containing all dem param tuples
        true = y_test[0]
        # get predictions from trained multioutput model
        pred = model.predict(X_test)
        # have to sort true and pred by param to plot results by param
        param_true, param_pred = sort_by_param(true, pred)

        # get scores for normal pred versions (logged)
        r2_all = get_r2(true, pred)[1]
        rho = get_rho(true, pred)

        for i, _ in enumerate(param_true):
            # handling log-scale data
            if i < len(logs) - 1 and logs[i]:  # misid is not included in log
                # convert log-scale values back to regular scale
                plot_p_true = [10**p_true for p_true in param_true[i]]
                plot_p_pred = [10**p_pred for p_pred in param_pred[i]]
                log = True
            else:  # leave as is if values not in log scale
                plot_p_true = param_true[i]
                plot_p_pred = param_pred[i]
                log = False
            # handling scores
            r2 = r2_all[i]
            rho = get_rho(plot_p_true, plot_p_pred)
            # set title
            if params is None:
                title = f'param {i + 1}'
            else:
                title = f'{params[i]}'
            if theta:
                title += f' theta {theta}'
            # plot a single subplot
            plot_accuracy_single(plot_p_true, plot_p_pred, size=[6, 2, 20],
                                 log=log, r2=r2, rho=rho, title=title)
            plt.savefig(f'{results_prefix}_param_{i + 1:02d}_accuracy')
            plt.clf()
