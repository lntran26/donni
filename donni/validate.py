import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, spearmanr
# from mapie.metrics import regression_coverage_score
from tensorflow import keras
import keras.backend as K
from donni.train import CustomLayer


def root_mean_squared_error(pred_pre: np.ndarray, true_pre: np.ndarray):
    # exclude nans
    pred_post = pred_pre[~np.isnan(pred_pre)]
    true_post = true_pre[~np.isnan(pred_pre)]
    return ((true_post - pred_post) ** 2).mean() ** 0.5


def regression_coverage_score(y_true,
                            y_pred_low,
                            y_pred_up):
    "Replacing dependency for mapie.metrics.regression_coverage_score"
    coverage = np.mean(
        ((y_pred_low <= y_true) & (y_pred_up >= y_true))
    )
    return float(coverage)


def get_coverage(pis_all_params, y_test, alpha):
    """Get coverage scores for mapie MLP models"""
    all_coverage = []
    for param_i, pis in enumerate(pis_all_params):
        true = y_test[param_i] # labels
        
        coverage_scores = [
            regression_coverage_score(true, pis[i, 0, :], pis[i, 1, :])
            for i, _ in enumerate(alpha)
        ]
        
        all_coverage.append(coverage_scores)
    return all_coverage
    
def plot_coverage(cov_scores, alpha, results_prefix, eps=None, params=None):
    """Helper method to plot coverage plot"""
    
    font = {"size": 22}
    plt.rc("font", **font)

    expected = [100 * (1 - a) for a in alpha]
    observed = []
    for cov_score in cov_scores:
        observed.append([s * 100 for s in cov_score])

    ax = plt.gca()
    ax.set_aspect("equal", "box")

    title = "C.I. coverage\n"
    if eps:
        title += f"eps={eps}"

    ax.text(0.05, 0.95, title, transform=ax.transAxes, va="top", fontsize=18)
    ax.set_xlabel("Expected", fontsize=20)
    ax.set_ylabel("Observed", fontsize=20)

    for i in range(len(cov_scores)):
        label = params[i] if params is not None else "param " + str(i + 1)
        ax.plot(expected, observed[i], label=label, linewidth=2)
    ax.plot([0, 100], [0, 100], "-k", zorder=-100, lw=1)
    # define ticks
    plt.xticks(ticks=list(range(0, 101, 25)))
    plt.yticks(ticks=list(range(0, 101, 25)))
    plt.tick_params(length=10, which="major")
    plt.rc("xtick", labelsize=22)
    plt.rc("ytick", labelsize=22)
    # define axis range
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    ax.legend(fontsize=15, 
              frameon=False,
              bbox_to_anchor=(1, 0), loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{results_prefix}_coverage.png", bbox_inches="tight")
    plt.clf()
    
def plot_accuracy_single(
    x,
    y,
    size=(8, 2, 20),
    x_label="Simulated",
    y_label="Inferred",
    log=False,
    r2=None,
    rho=None,
    rmse=None,
    c=None,
    title=None,
):
    """
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    rmse: rmse score for x and y
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    """
    font = {"size": size[2]}
    plt.rc("font", **font)
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect("equal", "box")

    # plot data points in a scatter plot
    if c is None:
        # 's' specifies dots size
        plt.scatter(x, y, s=size[0] * 2**3, alpha=0.8)
    else:  # condition to add color bar
        plt.scatter(x, y, c=c, vmax=5, s=size[0] * 2**3, alpha=0.8)
        # vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r"$\frac{T}{ν}$")

    # axis label texts
    plt.xlabel(x_label, labelpad=size[2] / 2)
    plt.ylabel(y_label, labelpad=size[2] / 2)

    # only plot in log scale if log specified for the param
    x=x.tolist()
    y=y.tolist()
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x + y) * 10**-0.5, max(x + y) * 10**0.5])
        plt.ylim([min(x + y) * 10**-0.5, max(x + y) * 10**0.5])
        plt.xticks(ticks=[1e-2, 1e0, 1e2])
        plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    else:
        # axis scales customized to data
        if max(x + y) > 1:
            plt.xlim([min(x + y) - 0.5, max(x + y) + 0.5])
            plt.ylim([min(x + y) - 0.5, max(x + y) + 0.5])
        else:
            plt.xlim([min(x + y) - 0.05, max(x + y) + 0.05])
            plt.ylim([min(x + y) - 0.05, max(x + y) + 0.05])
    plt.tick_params("both", length=size[2] / 2, which="major")

    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (1, 1), linewidth=size[1] / 2, color="black", zorder=-100)

    # plot scores if specified
    if r2 is not None:
        plt.text(
            0.25,
            0.82,
            "\n\n" + r"$R^{2}$: " + str(round(r2, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    if rho is not None:
        plt.text(
            0.25,
            0.82,
            "ρ: " + str(round(rho, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    if rmse is not None:
        plt.text(
            0.7,
            0.08,
            "rmse: " + str(round(rmse, 3)),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=size[2],
            transform=ax.transAxes,
        )
    if title is not None:
        ax.text(0.05, 0.98, title, transform=ax.transAxes, va="top")
    plt.tight_layout()

def plot_interval(int_arr, log, param, results_prefix, fontsize=12):

    x = range(1, int_arr.shape[1]+1)
    
    fig = plt.figure(figsize=(8, 2), dpi=300)
    ax = plt.gca()
    ax.margins(x=0.01)

    # color the range of true parameter value
    ax.fill_between(x, min(int_arr[0]), max(int_arr[0]), alpha=0.2)
    
    # only plot in log scale if log
    if log:
        plt.yscale("log")
        ax.set_ylim(1e-3, 1e3)
    else:
        offset = (max(int_arr[0]) - min(int_arr[0])) / 3
        ax.set_ylim(min(int_arr[0])-offset, max(int_arr[0])+offset)

    ax.tick_params('y', length=4, which='major', labelsize=fontsize)
    
    # ax.get_xaxis().set_visible(False) # whether to have x axis
    ax.tick_params(axis='x', labelsize=12)
        
    # show first and last value of x ticks only
    # ax.set_xticks([1, 15, 30, 45, 60, int_arr.shape[1]])
    x_first_tick = 1
    x_last_tick = int_arr.shape[1]
    x_major_ticks_count = 10
    ax.set_xticks(
    [int(x_first_tick + (x_last_tick - x_first_tick) * i / (x_major_ticks_count - 1)) for i in range(x_major_ticks_count)],
    minor=False)
        
    plt.minorticks_off()
    
    # plot true values
    ax.scatter(x, int_arr[0], s=3, zorder=2.5, alpha=0.6, marker='s')
    
    # plot prediction values and interval bars
    neg_int = int_arr[1] - int_arr[2]
    pos_int = int_arr[3] - int_arr[1]
    ax.errorbar(x, int_arr[1], 
                yerr=[
                abs(neg_int), abs(pos_int)], 
                # fmt='o',
                fmt='.',
                markersize='3',
                elinewidth=0.5, 
                label = 'inferred value, with interval', c="tab:brown")
    
    # figure title and legend location
    ax.set_title(f'{param}, 95% confidence interval', fontsize=fontsize)
    plt.savefig(f"{results_prefix}_{param}_95_CI.png", bbox_inches="tight")
    plt.clf()
    
def validate(filename_list, mlpr_dir, X_test, y_test, params, logs, plot_prefix):
    alpha=(0.05, 0.1, 0.2, 0.5, 0.7, 0.85)
    all_pis = []
    all_means = []
    all_vars = []
    for filename in filename_list:
        if filename.startswith("param") and filename.endswith("predictor.keras"):
            mlpr = keras.models.load_model(f'{mlpr_dir}/{filename}', 
                                    custom_objects={'CustomLayer': CustomLayer})
            # tentatively print model structure
            with open(plot_prefix + '_report.txt','a') as fh:
                # Pass the file handle in as a lambda function to make it callable
                mlpr.summary(print_fn=lambda x: fh.write(x + '\n'))
                # print model learning rate
                # print(f"Learning rate: {K.eval(mlpr.optimizer.lr)}", file=fh)
            mean, var = mlpr.predict(X_test)
            all_means.append(mean)
            all_vars.append(var)
            
        pis_per_param = []
        for a in alpha:
            z_score = round(norm.ppf(1-(a)/2), 2)
            lower = mean - z_score * np.sqrt(var)
            upper = mean + z_score * np.sqrt(var)
            pis = np.stack((lower, upper))
            pis_per_param.append(np.squeeze(pis))
        all_pis.append(pis_per_param)
    
    # plot coverage
    cov_scores = get_coverage(np.array(all_pis), np.array(y_test), alpha)
    plot_coverage(np.array(cov_scores), alpha, f"{plot_prefix}_coverage", params=params)
    
    # plot regular accuracy
    for i, param in enumerate(params):
        true = np.squeeze(np.array(y_test[i]))
        pred = np.squeeze(np.array(all_means[i]))
        if logs[i]:
            true = 10**true
            pred = 10**pred

        fig = plt.figure()
        plot_accuracy_single(true, 
                            pred,
                            log=logs[i],
                            rho=spearmanr(true, pred)[0],
                            rmse=root_mean_squared_error(np.array(pred), np.array(true)),
                            title=params[i])
        plt.savefig(f"{plot_prefix}_param_{i + 1:02d}_accuracy.png", bbox_inches="tight")
        plt.clf()
        
    # plot accuracy with 95% CI width
    for i, param in enumerate(params):
        true = np.squeeze(np.array(y_test[i]))
        pred = np.squeeze(np.array(all_means[i]))
        err = np.squeeze(np.sqrt(all_vars[i]))
        
        fig = plt.figure()
        ax = plt.gca()

        # plot accuracy
        plt.scatter(true, pred, 
                    s=5, zorder=100)
        # plot a line of slope 1 (perfect correlation)
        plt.axline((0, 0), (0.1, 0.1), linewidth=0.5, color="black", zorder=-100)
        # plot 95% confidence interval
        ax.errorbar(true, pred,
                yerr=1.96*err,
                fmt='.',
                markersize='3',
                elinewidth=0.5)
        # error score text
        rho=spearmanr(true, pred)[0]
        plt.text(   0.25,
                    0.82,
                    "ρ: " + str(round(rho, 3)),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
        plt.title(f'{param}')
        plt.savefig(f"{plot_prefix}_param_{i + 1:02d}_accuracy_95_ci.png", bbox_inches="tight")
        plt.clf()