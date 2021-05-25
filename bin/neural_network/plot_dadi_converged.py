'''
plots dadi vs nn comparisons
'''
import util
import pickle
import math
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # this list is of form [[true, pred], [], [], []]
    list_dadi_results = pickle.load(open('dadi_opt_results_converged', 'rb'))
    # pred is None if not converged (todo)
  
    theta_list = [1000]
    size = len(theta_list)
    fig1, axs1 = plt.subplots(1, size, figsize=(4*size, 4)) 
    fig2, axs2 = plt.subplots(1, size, figsize=(4*size, 4))
    fig3, axs3 = plt.subplots(1, size, figsize=(4*size, 4))
    fig4, axs4 = plt.subplots(1, size, figsize=(4*size, 4))
        
        
    i = 0
    for theta, true, pred in zip(theta_list, list_true, list_results):
        r2 = util.nn_r2_score(true, pred)[1]
        param_true, param_pred = util.sort_by_param(true, pred)
        util.plot_by_param_log(param_true[0], param_pred[0], True, axs1[i],
                                r2=r2[0], case=['nu1', theta_list[i]])
        util.plot_by_param_log(param_true[1], param_pred[1], True, axs2[i],
                                r2=r2[1], case=['nu2', theta_list[i]])
        util.plot_by_param_log(param_true[2], param_pred[2], False, axs3[i],
                                r2=r2[2], case=['T', theta_list[i]])
        util.plot_by_param_log(param_true[3], param_pred[3], False, axs4[i],
                                r2=r2[3], case=['m', theta_list[i]])
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
    
    