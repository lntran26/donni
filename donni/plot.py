"""Module for using trained MLPR to plot many demographic param predictions"""
import pickle
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from scipy import stats
import matplotlib.pyplot as plt


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


def plot_accuracy_single(x, y, size=(8, 2, 20), x_label="Simulated",
                         y_label="Inferred", log=False, r2=None,
                         rho=None, rmse=None, c=None, title=None):
    '''
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    rmse: rmse score for x and y
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    '''
    font = {'size': size[2]}
    plt.rc('font', **font)
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal', 'box')

    # plot data points in a scatter plot
    if c is None:
        plt.scatter(x, y, s=size[0]*2**3, alpha=0.8)  # 's' specifies dots size
    else:  # condition to add color bar
        plt.scatter(x, y, c=c, vmax=5, s=size[0]*2**3, alpha=0.8)
        # vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r'$\frac{T}{ν}$')

    # axis label texts
    plt.xlabel(x_label, labelpad=size[2]/2)
    plt.ylabel(y_label, labelpad=size[2]/2)

    # only plot in log scale if log specified for the param
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.ylim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.xticks(ticks=[1e-2, 1e0, 1e2])
        plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    else:
        # axis scales customized to data
        if max(x+y) > 1:
            plt.xlim([min(x+y)-0.5, max(x+y)+0.5])
            plt.ylim([min(x+y)-0.5, max(x+y)+0.5])
        else:
            plt.xlim([min(x+y)-0.05, max(x+y)+0.05])
            plt.ylim([min(x+y)-0.05, max(x+y)+0.05])
    plt.tick_params('both', length=size[2]/2, which='major')

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (1, 1), linewidth=size[1]/2, color='black', zorder=-100)

    # plot scores if specified
    if r2 is not None:
        plt.text(0.25, 0.82, "\n\n" + r'$R^{2}$: ' + str(round(r2, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes)
    if rho is not None:
        plt.text(0.25, 0.82, "ρ: " + str(round(rho, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes)
    if rmse is not None:
        plt.text(0.7, 0.08, "rmse: " + str(round(rmse, 3)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if title is not None:
        ax.text(0.05, 0.98, title, transform=ax.transAxes, va='top')
    plt.tight_layout()


def plot_coverage(cov_scores, alpha, theta=None, params=None):
    """Helper method to plot coverage plot"""

    expected = [100*(1 - a) for a in alpha]
    observed = []
    for cov_score in cov_scores:
        observed.append([s*100 for s in cov_score])

    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    title = 'C.I. coverage\n'
    if theta:
        title += f'θ={theta}'

    # ax.set_title(title, fontsize=15)
    ax.text(0.05, 0.95, title, transform=ax.transAxes, va='top', fontsize=18)
    ax.set_xlabel('Expected', fontsize=20)
    ax.set_ylabel('Observed', fontsize=20)

    for i in range(len(cov_scores)):
        label = params[i] if params is not None else 'param '+str(i+1)
        ax.plot(expected, observed[i], label=label, linewidth=2)
    ax.plot([0, 100], [0, 100], '-k', zorder=-100, lw=1)
    # define ticks
    plt.xticks(ticks=list(range(0, 101, 25)))
    plt.yticks(ticks=list(range(0, 101, 25)))
    plt.tick_params(length=10, which='major')
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    # define axis range
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    ax.legend(fontsize=15, frameon=False,
              bbox_to_anchor=(1, 0), loc="lower left")
    plt.tight_layout()


def process_inference(raw_true, raw_pred, log):
    """
    Convert log params and get accuracy scores
    Input:
        raw prediction and true values
        log: whether param is in log scale
    """
    # log transform
    if log:
        true = [10**p_true for p_true in raw_true]
        pred = [10**p_pred for p_pred in raw_pred]
    else:
        true = list(raw_true)
        pred = list(raw_pred)
    # get scores
    r2 = r2_score(true, pred)
    rho = stats.spearmanr(true, pred)[0]
    rmse = mean_squared_error(true, pred, squared=False)

    return true, pred, r2, rho, rmse


def get_coverage(models: list, X_test, y_test, alpha):
    """Get coverage scores for mapie MLP models"""
    all_coverage = []
    for model_i, model in enumerate(models):
        true = y_test[model_i]
        _, pis = model.predict(X_test, alpha=list(alpha))
        coverage_scores = [
            regression_coverage_score(true, pis[:, 0, i], pis[:, 1, i])
            for i, _ in enumerate(alpha)
        ]
        all_coverage.append(coverage_scores)
    return all_coverage


def get_title(params, i, theta):
    """Get title for plot from param name and theta"""
    if params is None:
        title = f'param {i + 1}'
    else:
        title = f'{params[i]}'
    if theta:
        title += f' θ={theta}'
    return title


def retrain(mlpr, i, X_input, label, count, reason, mlpr_dir, results_prefix):
    """Method for retraining one mapie MLPR"""
    # open QC text file to record information
    with open(f'{results_prefix}_QC.txt', 'a') as fh:
        fh.write(f'Retraining param {i+1:02d}: ')
        fh.write(f'Retrain {count} Reason: {reason}\n')

    # get current mlpr hyperparam
    mlpr_spec = mlpr.__dict__['estimator'].get_params()
    # initialize a new sklearn mlpr
    mlpr_init = MLPRegressor()
    # load mlpr hyperparam
    mlpr_init.set_params(**mlpr_spec)
    # wrap sklearn mlpr into mapie mlpr
    new_mlpr = MapieRegressor(mlpr_init)
    # train new mlpr
    new_mlpr.fit(X_input, label)
    # save new mlpr
    pickle.dump(new_mlpr, open(
                f'{mlpr_dir}/param_{i+1:02d}_predictor', 'wb'))

    return new_mlpr


def plot(models: list, X_test, y_test, X_input, y_label, results_prefix,
         mlpr_dir, logs, mapie=True, coverage=False, theta=None, params=None,
         alpha=(.05, .1, .2, .5, .7, .85)):
    """Main method to plot both accuracy and coverage plots"""

    # make prediction with trained mlpr models
    if mapie:
        for i, (model, log) in enumerate(zip(models, logs)):
            # internal QC per trained MLP
            rerun_count = 0
            while rerun_count < 10:  # limit retrain to 10 times
                # get prediction and accuracy score
                raw_pred = model.predict(X_test)
                true, pred, r2, rho, rmse = process_inference(y_test[i],
                                                              raw_pred, log)
                if rho is np.nan:
                    rerun_count += 1
                    model = retrain(model, i, X_input, y_label[i], rerun_count,
                                    'rho is nan', mlpr_dir, results_prefix)
                elif rho <= 0.05:
                    rerun_count += 1
                    model = retrain(model, i, X_input, y_label[i], rerun_count,
                                    'rho <= 0.05', mlpr_dir, results_prefix)
                elif (log and max(pred) > max(true)*1e4 or
                      log and min(pred) < min(true)/1e4):
                    rerun_count += 1
                    model = retrain(model, i, X_input, y_label[i], rerun_count,
                                    'log param inferred 4 logs out of bounds',
                                    mlpr_dir, results_prefix)
                elif (not log and max(pred) > max(true)*10 or
                      not log and abs(min(pred)) > max(true)*10):
                    rerun_count += 1
                    model = retrain(model, i, X_input, y_label[i], rerun_count,
                                    'param inferred 10x out of bounds',
                                    mlpr_dir, results_prefix)
                else:
                    break

            title = get_title(params, i, theta)
            plot_accuracy_single(true, pred, size=[6, 2, 20], log=log,
                                 r2=r2, rho=rho, rmse=rmse, title=title)
            plt.savefig(f'{results_prefix}_param_{i + 1:02d}_accuracy',
                        bbox_inches='tight')
            plt.clf()

        if coverage:
            all_coverage = get_coverage(models, X_test, y_test, alpha)
            plot_coverage(all_coverage, alpha, theta, params)
            plt.savefig(f'{results_prefix}_coverage', bbox_inches='tight')
            plt.clf()

    else:
        # for sklearn multioutput, models is a list of one mlpr
        model = models[0]
        # and true is a list of one list containing all dem param tuples
        true = y_test[0]
        # get predictions from trained multioutput model
        pred = model.predict(X_test)
        # have to sort true and pred by param to plot results by param
        sorted_true, sorted_pred = sort_by_param(true, pred)

        for i, (p_true, p_pred, log) in enumerate(zip(sorted_true, sorted_pred,
                                                      logs)):
            true, pred, r2, rho, rmse = process_inference(p_true, p_pred, log)
            title = get_title(params, i, theta)
            plot_accuracy_single(true, pred, size=[6, 2, 20], log=log,
                                 r2=r2, rho=rho, rmse=rmse, title=title)
            plt.savefig(f'{results_prefix}_param_{i + 1:02d}_accuracy',
                        bbox_inches='tight')
            plt.clf()
