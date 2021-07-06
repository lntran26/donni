import matplotlib.pyplot as plt
import ml_models, data_manip
from scipy import stats

def plot_by_param_4x4(true,pred,log=False,r2=None,msle=None,rho=None,c=None):
    # TO DO: add arg for dot size, line width to make 2x2 and 4x4 the same fn
    # also text font size
    '''
    Plot a single true vs. predict panel for one train:test pair
    true, pred = list of true and predicted values for one param,
    which can be obtained from sort_by_param;
    r2: one r2 score for one param of one train:test pair
    msle: one msle score for one param of one train:test pair
    '''
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal','box')
    if c is None:
        plt.scatter(true, pred, s=8*2**3) # 's' to change dots size
    else:
        plt.scatter(true, pred, c=c, vmax=5, s=8*2**3) #vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r'$\frac{T}{ν}$', fontweight='bold', fontsize=20)
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
    plt.axline((0,0),(1,1),linewidth=2)
    if r2 != None:
        plt.text(0.4, 0.9, r'$R^{2}$: ' + str(round(r2,4)), horizontalalignment='center', verticalalignment='center', fontsize=20,
        transform = ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "\n\nρ: " + str(round(rho,4)), horizontalalignment='center', verticalalignment='center', fontsize=20,transform = ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle,4)), horizontalalignment='center', verticalalignment='center', fontsize=20,transform = ax.transAxes)        

# plotting function with specific settings for 2x2 plot
def plot_by_param_2x2(true,pred,log=False,r2=None,msle=None,rho=None,c=None):
    '''
    true, pred = list of true and predicted values for one param,
    which can be obtained from sort_by_param;
    r2: one r2 score for one param of one train:test pair
    msle: one msle score for one param of one train:test pair
    '''
    # assign ax variable to customize axes
    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal','box')
    if c is None:
        plt.scatter(true, pred, s=20*2**3) # 's' to change dots size
    else:
        plt.scatter(true, pred, c=c, vmax=5, s=20*2**3) #vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        # cbar.ax.zorder = -1
        cbar.ax.set_title(r'$\frac{T}{ν}$', fontweight='bold', fontsize=50)
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
    plt.axline((0,0),(1,1),linewidth=2)

    if r2 != None:
        plt.text(0.4, 0.9, r'$R^{2}$: ' + str(round(r2,4)), horizontalalignment='center', verticalalignment='center', fontsize=40,
        transform = ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "\n\nρ: " + str(round(rho,4)), horizontalalignment='center', verticalalignment='center', fontsize=40,transform = ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle,4)), horizontalalignment='center', verticalalignment='center', fontsize=40,transform = ax.transAxes)  

def plot_accuracy_single(
    true, pred,size,log=False,r2=None,msle=None,rho=None,c=None):
    # TO DO: add arg for dot size, line width to make 2x2 and 4x4 the same fn
    # also text font size
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
        plt.scatter(true, pred, c=c, vmax=5, s=8*2**3) #vmax: colorbar limit
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
        plt.text(0.4, 0.9, r'$R^{2}$: ' + str(round(r2,4)), horizontalalignment='center', verticalalignment='center', 
        fontsize=size[2], transform = ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "\n\nρ: " + str(round(rho,4)), horizontalalignment='center', verticalalignment='center', 
        fontsize=size[2],transform = ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle,4)), horizontalalignment='center', verticalalignment='center', 
        fontsize=size[2],transform = ax.transAxes)  

def plot_accuracy_multi(
    list_models, list_test_dict, params, model_name, logs, size):
    # list_models = list_mlpr or list_rfr
    # params = ['s', r'$ν_1$', r'$ν_2$', 'T', 'm12', 'm21']
    # models = [' RFR',' MLPR']
    # logs = [False, True, True, False, False, False]
    # size = ((title_font_size, title_pad), (axis_font_size, axis_pad), single_plot_size_tuple, fig_size)
    # size = ((30, 20), (20, 60),(8,2,20))
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

    # testing, and plotting
    count_pos = 1
    for model in reversed(list_models):# flip the order of variance for plotting
        for test_dict in list_test_dict:
            y_true, y_pred = ml_models.model_test(model, test_dict)
            # sort test results by param for plotting
            param_true, param_pred = data_manip.sort_by_param(y_true, y_pred)
            # calculate r2 and rho scores by param
            r2_by_param = ml_models.r2(y_true, y_pred)[1]
            rho_by_param = stats.spearmanr(y_true, y_pred)
            
            # PLOT MULTIPLE SUBPLOT
            for i in range(len(params)):
                plt.figure(i+1).add_subplot(len(list_models), 
                                            len(list_test_dict), count_pos)
                if logs[i]:
                    log_p_true = [10**p_true for p_true in param_true[i]]
                    log_p_pred = [10**p_pred for p_pred in param_pred[i]]
                    plot_accuracy_single(log_p_true, log_p_pred, size[2],    
                                    log=True,
                                    r2=r2_by_param[i],
                                    rho=rho_by_param[0][i][i+len(params)])
                else:
                    plot_accuracy_single(param_true[i], param_pred[i], 
                                    size[2],
                                    r2=r2_by_param[i],
                                    rho=rho_by_param[0][i][i+len(params)])
            count_pos += 1