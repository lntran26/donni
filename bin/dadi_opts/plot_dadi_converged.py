'''
plots dadi vs nn comparisons
'''
import util
import pickle
import math
import matplotlib.pyplot as plt
import time
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import numpy as np

if __name__ == "__main__":
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # this list is of form [[true, pred], [], [], []], length 50
    list_dadi_results = pickle.load(open('dadi_opt_results_converged', 'rb'))
    # pred is None if not converged (todo)

    list_true, list_pred, list_true_moff, list_pred_moff = [], [], [], []

    # create the lists
    # if m pred is off by < 1, add to the normal true/pred lists for plotting (log transform nus)
    # if m pred is off by >= 1, add that set to the "moff" lists and don't log transform
    for x in list_dadi_results:
        if x[1] is not None:
            if abs(x[1][3] - x[0][3]) < 1:
                list_true.append([math.log(x[0][0], 10), math.log(x[0][1], 10), x[0][2], x[0][3]])
                list_pred.append([math.log(x[1][0], 10), math.log(x[1][1], 10), x[1][2], x[1][3]])
            else:
                list_true_moff.append(x[0])
                list_pred_moff.append(x[1])
    

    print(len(list_true))       # 19
    print(len(list_true_moff))  # 16

##    # see the non-off params
##    i = 0
##    for true, pred in zip(list_true, list_pred): 
##        print(f"#{i+1}--true: {true}, pred: {pred}")
##        i += 1
##    print()

    # print the params with m off by >= 1 and create ratios lists for plotting
    i = 0
    m_ratios, nu1_ratios, nu2_ratios = [], [], []
    for true, pred in zip(list_true_moff, list_pred_moff):
        t = np.array(true)
        rounded_t = np.around(t, 4)
        p = np.array(pred)
        rounded_p = np.around(p, 4)
        print(f"#{i+1}--true: {rounded_t}\t\tpred: {rounded_p}")
        m_ratio = pred[3]/true[3]
        nu1_ratio = true[0]/pred[0]
        nu2_ratio = true[1]/pred[1]
        m_ratios.append(m_ratio)
        nu1_ratios.append(nu1_ratio)
        nu2_ratios.append(nu2_ratio)
        print(f"    Ratios: m: {m_ratio:.4f}, nu1: {nu1_ratio:.4f}, nu2: {nu2_ratio:.4f}")
        i += 1
    # get r^2 scores and plot
    r2 = r2_score(m_ratios, nu1_ratios)
    ax = plt.gca()
    plt.plot([0, 10], [0, 10])
    plt.scatter(m_ratios, nu1_ratios)
    plt.xlabel("m_pred : m_true")
    plt.ylabel("nu1_true : nu1_pred")
    plt.text(0.3, 0.9, "R^2: " + str(round(r2, 4)), fontsize=16,
            horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    plt.show()

    
    theta_list = [1000]
    size = len(theta_list)
    fig1, axs1 = plt.subplots(1, size, figsize=(4*size, 4)) 
    fig2, axs2 = plt.subplots(1, size, figsize=(4*size, 4))
    fig3, axs3 = plt.subplots(1, size, figsize=(4*size, 4))
    fig4, axs4 = plt.subplots(1, size, figsize=(4*size, 4))
        
    # plot true vs. pred (same as nn)
    i = 0
    for theta, true, pred in zip(theta_list, [list_true], [list_pred]):
        r2 = util.nn_r2_score(true, pred)[1]
        param_true, param_pred = util.sort_by_param(true, pred)
        rhos = [spearmanr(param_true[i], param_pred[i])[0] for i in range(4)]
        util.plot_by_param_log(param_true[0], param_pred[0], True, axs1,
                                r2=r2[0], rho=rhos[0], case=['nu1', theta_list[i]])
        util.plot_by_param_log(param_true[1], param_pred[1], True, axs2,
                                r2=r2[1], rho=rhos[1], case=['nu2', theta_list[i]])
        util.plot_by_param_log(param_true[2], param_pred[2], False, axs3,
                                r2=r2[2], rho=rhos[2], case=['T', theta_list[i]])
        util.plot_by_param_log(param_true[3], param_pred[3], False, axs4,
                                r2=r2[3], rho=rhos[3], case=['m', theta_list[i]])
        i += 1
    
    fig1.tight_layout(rect=[0, 0, 1, 0.95])    
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig4.tight_layout(rect=[0, 0, 1, 0.95]) 

    fig1.savefig(f'../../results/{timestr}_splitmig_dadi_nu1.png')
    fig2.savefig(f'../../results/{timestr}_splitmig_dadi_nu2.png')
    fig3.savefig(f'../../results/{timestr}_splitmig_dadi_T.png')
    fig4.savefig(f'../../results/{timestr}_splitmig_dadi_m.png')

    print("END")
