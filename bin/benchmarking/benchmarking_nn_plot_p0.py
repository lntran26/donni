import pickle
import dadi
import numpy as np
import util
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__': 
    # import test set files to plot and check input prediction
    test_set_1 = pickle.load(open(
        'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_4_bounds','rb'))
    test_set_2 = pickle.load(open(
        'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_5_bounds','rb'))
    test_set_3 = pickle.load(open(
        'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_6_bounds','rb'))

    # # combine all results into one set
    # test_set = test_set_1 + test_set_2 + test_set_3
    
    # each test set is a list of length 60 tuples
    # each tuple is (p_true, fs, p0)
    # where p0 is 20 dadi only, 20 RFR_1, 20 avg_RFR4
    # p_true and fs are the same 20 set repeated 3 times

    def extract_p(test_set):
        p_true, p0, p1, p2 = [], [], [], []
        for i in range(20):
            p_true.append(test_set[i][0])
            p0.append(test_set[i][2])
        for i in range(20,40):
            p1.append(test_set[i][2])
        for i in range(40,60):
            p2.append(test_set[i][2])
        return p_true, p0, p1, p2

    p_true_1, p0_1, p1_1, p2_1 = extract_p(test_set_1)
    p_true_2, p0_2, p1_2, p2_2 = extract_p(test_set_2)
    p_true_3, p0_3, p1_3, p2_3 = extract_p(test_set_3)

    y_true = p_true_1 + p_true_2 + p_true_3
    
    y_pred_1 = p0_1 + p0_2 + p0_3
    y_pred_2 = p1_1 + p1_2 + p1_3
    y_pred_3 = p2_1 + p2_2 + p2_3
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
    plt.ylabel('NN_1000')
    plt.subplot(3, 4, 9)
    plt.ylabel('NN_10000')

    fig.savefig('results/2d-splitmig/benchmarking_nn/benchmarking_nn_all_p0_Spearman.png', bbox_inches='tight')
