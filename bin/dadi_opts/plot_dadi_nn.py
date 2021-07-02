'''
plots dadi vs nn comparisons
'''
import util
import pickle
import math
import matplotlib.pyplot as plt
import time
from scipy import stats

if __name__ == "__main__":
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    '''
    # this list is of form [[pred_params], [], [], []] such that index 0 is tested on theta 1
    list_dadi_results = pickle.load(open('dadi_parallel_results', 'rb'))
    
    # this list is of form [(true, pred), (), (), ()...] len 16, where 0-4 is trained on theta 1
    list_nn_results = pickle.load(open('nn_results', 'rb'))
    '''
    dadi_results = pickle.load(open('../../results/20210419-212358_dadi_opt_full_results', 'rb'))
    
    # one row, four cols for each theta case
    #theta_list = [1, 100, 1000, 1000]
    theta_list = [1000]
    size = len(theta_list)
    fig1, axs1 = plt.subplots(size, size, figsize=(4*size, 4*size)) 
    fig2, axs2 = plt.subplots(size, size, figsize=(4*size, 4*size))
    fig3, axs3 = plt.subplots(size, size, figsize=(4*size, 4*size))
    fig4, axs4 = plt.subplots(size, size, figsize=(4*size, 4*size))
    
    results = [] # dadi results
    for i in range(len(dadi_results)):
        results.append([math.log(dadi_results[i][0][0], 10), 
                            math.log(dadi_results[i][0][1], 10),
                            dadi_results[i][0][2],
                            dadi_results[i][0][3]])
    
    # get nn results for same dataset
    list_train_dict = pickle.load(open('train_set','rb'))
    train_dict = list_train_dict[2] # 1000
    nn = util.nn_train(train_dict)
    fs_test_list = pickle.load(open('dadi_fs_data', 'rb'))
    nn_results = []
    for fs in fs_test_list:
        test_fs = fs/fs.sum()
        test_fs = test_fs.flatten()
        nn_results.append(nn.predict([test_fs]).flatten())
    
    theta = 1000
    r2 = util.nn_r2_score(results, nn_results)[1]
    param_dadi, param_nn = util.sort_by_param(results, nn_results)
    util.plot_by_param_log(param_dadi[0], param_nn[0], True, axs1,
                             r2=r2[0], case=['nu1', theta])
    util.plot_by_param_log(param_dadi[1], param_nn[1], True, axs2,
                            r2=r2[1], case=['nu2', theta])
    util.plot_by_param_log(param_dadi[2], param_nn[2], False, axs3,
                            r2=r2[2], case=['T', theta])
    util.plot_by_param_log(param_dadi[3], param_nn[3], False, axs4,
                             r2=r2[3], case=['m', theta])
    
    '''
    list_results = []
    for i in range(4):
        results = []
        for j in range(100):
            results.append([math.log(list_dadi_results[i][j][0][0], 10), 
                                        math.log(list_dadi_results[i][j][0][1], 10),
                                        list_dadi_results[i][j][0][2],
                                        list_dadi_results[i][j][0][3]])
        list_results.append(results)
        
        
    for train_i in range(4):
        list_nn_pred = [list(list_nn_results[i+4*train_i][1]) for i in range(4)]
        i = 0
        for theta, dadi, nn in zip(theta_list, list_results, list_nn_pred):
            r2 = util.nn_r2_score(dadi, nn)[1]
            param_true, param_pred = util.sort_by_param(dadi, nn)
            rhos = [stats.spearmanr(param_true[i], param_pred[i])[0] for i in range(4)]
            param_dadi, param_nn = util.sort_by_param(dadi, nn)
            util.plot_by_param_log(param_dadi[0], param_nn[0], True, axs1[train_i, i],
                                    r2=r2[0], rho=rhos[0], case=['nu1', theta_list[i]])
            util.plot_by_param_log(param_dadi[1], param_nn[1], True, axs2[train_i, i],
                                    r2=r2[1], rho=rhos[1], case=['nu2', theta_list[i]])
            util.plot_by_param_log(param_dadi[2], param_nn[2], False, axs3[train_i, i],
                                    r2=r2[2], rho=rhos[2], case=['T', theta_list[i]])
            util.plot_by_param_log(param_dadi[3], param_nn[3], False, axs4[train_i, i],
                                    r2=r2[3], rho=rhos[3], case=['m', theta_list[i]])
            i += 1
    '''
    
    fig1.tight_layout(rect=[0, 0, 1, 0.95])    
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig3.tight_layout(rect=[0, 0, 1, 0.95]) 
    fig4.tight_layout(rect=[0, 0, 1, 0.95]) 

    fig1.savefig(f'../../results/{timestr}_splitmig_dadi_nn_nu1.png')
    fig2.savefig(f'../../results/{timestr}_splitmig_dadi_nn_nu2.png')
    fig3.savefig(f'../../results/{timestr}_splitmig_dadi_nn_T.png')
    fig4.savefig(f'../../results/{timestr}_splitmig_dadi_nn_m.png')

    print("END")
    
    