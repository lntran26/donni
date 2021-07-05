import matplotlib.pyplot as plt

def plot_by_param_4x4(true,pred,r2=None,msle=None,rho=None,c=None):
    '''
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

    # only plot in log scale if the difference between max and min is large
    if max(true+pred)/min(true+pred) > 500:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(true+pred)*10**-0.5, max(true+pred)*10**0.5])
        plt.ylim([min(true+pred)*10**-0.5, max(true+pred)*10**0.5])
        # plot a slope 1 line
        plt.plot([10**-5, 10**3], [10**-5, 10**3], linewidth=2)
    else:
        # axis scales customized to data
        plt.xlim([min(true+pred)-0.5, max(true+pred)+0.5])
        plt.ylim([min(true+pred)-0.5, max(true+pred)+0.5])
        # plot a slope 1 line
        plt.plot([-2, 13], [-2, 13], linewidth=2)

    if r2 != None:
        plt.text(0.4, 0.9, r'$R^{2}$: ' + str(round(r2,4)), horizontalalignment='center', verticalalignment='center', fontsize=20,
        transform = ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "\n\nρ: " + str(round(rho,4)), horizontalalignment='center', verticalalignment='center', fontsize=20,transform = ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle,4)), horizontalalignment='center', verticalalignment='center', fontsize=20,transform = ax.transAxes)        

# plotting function with specific settings for 2x2 plot
def plot_by_param_2x2(true,pred,r2=None,msle=None,rho=None,c=None):
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

    # only plot in log scale if the difference between max and min is large
    if max(true+pred)/min(true+pred) > 500:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(true+pred)*10**-0.5, max(true+pred)*10**0.5])
        plt.ylim([min(true+pred)*10**-0.5, max(true+pred)*10**0.5])
        # plot a slope 1 line
        plt.plot([10**-5, 10**3], [10**-5, 10**3], linewidth=4)
    else:
        # axis scales customized to data
        plt.xlim([min(true+pred)-0.5, max(true+pred)+0.5])
        plt.ylim([min(true+pred)-0.5, max(true+pred)+0.5])
        # plot a slope 1 line
        plt.plot([-2, 13], [-2, 13], linewidth=4)
    if r2 != None:
        plt.text(0.4, 0.9, r'$R^{2}$: ' + str(round(r2,4)), horizontalalignment='center', verticalalignment='center', fontsize=40,
        transform = ax.transAxes)
    if rho != None:
        plt.text(0.4, 0.9, "\n\nρ: " + str(round(rho,4)), horizontalalignment='center', verticalalignment='center', fontsize=40,transform = ax.transAxes)
    if msle != None:
        plt.text(0.4, 0.9, "\n\n\n\nMSLE: " + str(round(msle,4)), horizontalalignment='center', verticalalignment='center', fontsize=40,transform = ax.transAxes)  

# ! log-scale param, define log
def plot_by_param_log(true,pred,log, ax, r2=None, case=None, vals=None):
    # case is of the form (name,  test_theta_val)
    ax_min = min(min(true), min(pred)) - 0.1
    ax_max = max(max(true), max(pred)) + 0.1
    if log:
        ax_min = 10**ax_min
        ax_max = 10**ax_max
        true = [10**num for num in true]
        pred = [10**num for num in pred]
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.plot([ax_min, ax_max], [ax_min, ax_max])
    if r2:
        x, y = ax_min + 0.1, ax_max - 0.1
        if log:
            x = 10**(math.log(ax_min, 10) + 0.1)
            y = 10**(math.log(ax_max, 10) - 0.1)
        ax.text(x, y, "R^2: " + str(round(r2, 3)))
    ax.set_title(f"{case[0]}: test theta {case[1]}")
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    ax.scatter(true, pred, c=vals)