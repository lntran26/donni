import pickle
import dadi
import numpy as np
import util
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__': 
    # import result files from hpc and test files
    # test files are just to check to make sure results are expected
    # test_set = pickle.load(open('benchmarking_test_set','rb'))
    # test_set_2 = pickle.load(open('benchmarking_test_set_2','rb'))
    # test_set_3 = pickle.load(open('benchmarking_test_set_3','rb'))

    result_set = pickle.load(open(
        'data/2d-splitmig/benchmarking_corrected_fs/normalized/benchmarking_results_set_no_perturb_3','rb'))

    # result_set_1 = pickle.load(open(
    #     'data/2d-splitmig/benchmarking_corrected_fs/normalized/benchmarking_results_set_no_perturb_1','rb'))
    # result_set_2 = pickle.load(open(
    #     'data/2d-splitmig/benchmarking_corrected_fs/normalized/benchmarking_results_set_no_perturb_2','rb'))
    # result_set_3 = pickle.load(open(
    #     'data/2d-splitmig/benchmarking_corrected_fs/normalized/benchmarking_results_set_no_perturb_3','rb'))

    # result_set = result_set_1 + result_set_2 + result_set_3
    y_true = []
    y_pred_1 = []
    y_pred_2 = []
    y_pred_3 = []
    for result in result_set:
        y_true.append(result[0])
        y_pred_1.append(result[1][0].tolist())
        y_pred_2.append(result[2][0].tolist())
        y_pred_3.append(result[3][0].tolist())
    y_pred = [y_pred_1, y_pred_2, y_pred_3]

    # Create four figures for each of the four param
    fig=plt.figure(1, figsize=(20,12), dpi=300)
    plt.axis("off")

    count_pos = 1
    for pred in y_pred: # for 3 cases
        param_true, param_pred = util.sort_by_param(y_true, pred)
        # using Spearman rho instead of Pearson's coefficient
        rho_by_param = stats.spearmanr(y_true, pred)

        for i in range(4):
            plt.figure(1)
            fig.add_subplot(3, 4, count_pos)
            util.plot_by_param(param_true[i], param_pred[i], rho=rho_by_param[0][i][i+4])
            count_pos += 1

    plt.figure(1)
    plt.subplot(3, 4, 1)
    plt.title('nu1')
    plt.ylabel('dadi only')
    plt.subplot(3, 4, 2)
    plt.title('nu2')
    plt.subplot(3, 4, 3)
    plt.title('T')
    plt.subplot(3, 4, 4)
    plt.title('m')
    plt.subplot(3, 4, 5)
    plt.ylabel('dadi + RFR_1')
    plt.subplot(3, 4, 9)
    plt.ylabel('dadi + RFR_avg')

    fig.savefig('results/2d-splitmig/benchmarking_corrected_fs/benchmarking_no_perturb_3_Spearman.png', bbox_inches='tight')