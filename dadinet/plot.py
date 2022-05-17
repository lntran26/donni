"""Module for using trained MLPR to plot many demographic param predictions"""
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score
import numpy as np
import dadi
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt


def get_r2(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    score_by_param = r2_score(y_true, y_pred, multioutput='raw_values')
    return score, score_by_param


def get_rho(y_true, y_pred):
    """stats.spearmanr returns two values: correlation and p-value
    Here we only want the correlation value"""
    return stats.spearmanr(y_true, y_pred)[0]


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
    if title != None:
        ax.set_title(title, fontsize=size[2], fontweight='bold')


def plot_coverage(cov_scores, alpha, results_prefix, theta=None):
    expected = [100*(1 - a) for a in alpha]
    observed = []
    for cov_score in cov_scores:
        observed.append([s*100 for s in cov_score])

    fig = plt.figure()
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
        ax.plot(expected, observed[i],
                label='param '+str(i+1), marker='o', linewidth=2)
    ax.plot(expected, expected, label='match', linewidth=2, color="black")

    plt.xticks(np.arange(min(expected), max(expected)+5, 10))
    plt.yticks(np.arange(min(expected), max(expected)+5, 10))
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    ax.legend()

    plt.savefig(f'{results_prefix}_coverage')
    plt.clf()


def plot(models: list, test_data, results_prefix, logs, mapie=True, coverage=False, theta=None):
    # unpack test_data dict
    X_test = [np.array(fs).flatten() for fs in test_data.values()]
    y_test = list(test_data.keys())

    # some set alpha
    alpha = [.05, .1, .2, .5, .7, .85]

    # parse labels into single list for each param (required for mapie)
    y_test_unpack = list(zip(*y_test)) if mapie else [y_test]
    # make prediction with trained mlpr models
    c = None
    if mapie:
        all_coverage = []
        c = None
        if len(logs) == 2:  # assumption for now
            T_true = y_test_unpack[1]
            nu_true = y_test_unpack[0]
            nu_true_delog = [10**nu for nu in nu_true]
            c = [T/nu for T, nu in zip(T_true, nu_true_delog)]
        for model_i, model in enumerate(models):
            true = y_test_unpack[model_i]
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
            title = f'param {model_i + 1}'
            if theta:
                title += f' theta {theta}'
            if model_i < len(logs) - 1 and logs[model_i]:
                # for log params, exponentiate each of the test and pred values
                true_delog = [10**p_true for p_true in true]
                pred_delog = [10**p_pred for p_pred in pred]
                plot_accuracy_single(true_delog, pred_delog, size=[6, 2, 20],
                                     log=True, r2=r2, rho=rho, title=title, c=c)
            else:
                true = [p_true for p_true in true]
                pred = [p_pred for p_pred in pred]
                plot_accuracy_single(true, pred, size=[6, 2, 20], log=False,
                                     r2=r2, rho=rho, title=title, c=c)
            plt.savefig(f'{results_prefix}_param_{model_i + 1}_accuracy')
            plt.clf()

        if coverage:
            # plot coverage
            plot_coverage(all_coverage, alpha, results_prefix)

    else:  # TODO: implement sklearn version
        return
