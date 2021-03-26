import pickle
import dadi
import numpy as np
import util
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == '__main__': 
    # load test set that contains the random forest prediction
    test_set = pickle.load(open('data/2d-splitmig/benchmarking/benchmarking_test_set','rb'))
    test_set_2 = pickle.load(open('data/2d-splitmig/benchmarking/benchmarking_test_set_2','rb'))
    test_set_3 = pickle.load(open('data/2d-splitmig/benchmarking/benchmarking_test_set_3','rb'))
    # each test set is a list of length 60 tuples
    # each tuple is (p_true, fs, p0)
    # where p0 is 20 dadi only, 20 RFR_1, 20 avg_RFR4
    # p_true and fs are the same 20 set repeated 3 times

    # process test_set to get the starting points for RFR_1 and avg_RFR4
    # def extract_p0(test_set):
    #     p1, p2 = [], []
    #     for i in range(20,40):
    #         p1.append(test_set[i][2])
    #     for i in range(40,60):
    #         p2.append(test_set[i][2])
    #     return p1, p2

    # p1_1, p2_1 = extract_p0(test_set)
    # p1_2, p2_2 = extract_p0(test_set_2)
    # p1_3, p2_3 = extract_p0(test_set_3)

    def extract_p0(test_set):
        p0, p1, p2 = [], [], []
        for i in range(20):
            p0.append(test_set[i][2])
        for i in range(20,40):
            p1.append(test_set[i][2])
        for i in range(40,60):
            p2.append(test_set[i][2])
        return p0, p1, p2

    p0_1, p1_1, p2_1 = extract_p0(test_set)
    p0_2, p1_2, p2_2 = extract_p0(test_set_2)
    p0_3, p1_3, p2_3 = extract_p0(test_set_3)

    result_set_1 = pickle.load(open(
        'data/2d-splitmig/benchmarking/benchmarking_results_set_no_perturb','rb'))
    result_set_2 = pickle.load(open(
        'data/2d-splitmig/benchmarking/benchmarking_results_set_no_perturb_2','rb'))
    result_set_3 = pickle.load(open(
        'data/2d-splitmig/benchmarking/benchmarking_results_set_no_perturb_3','rb'))

    # each result_set is a list of length 20 tuples
    # each tuple has the format (p_true, p1_opt, p2_opt, p3_opt)
    # where p1_opt, p2_opt, p3_opt is a tuple of params[] and LL

    result_set = result_set_1 + result_set_2 + result_set_3
    y_true = []
    # y_pred_1 = []
    # y_pred_2 = []
    # y_pred_3 = []
    y_pred_1 = p0_1 + p0_2 + p0_3
    y_pred_2 = p1_1 + p1_2 + p1_3
    y_pred_3 = p2_1 + p2_2 + p2_3
    for result in result_set:
        y_true.append(result[0])
        # y_pred_1.append(result[1][0].tolist())
        # y_pred_2.append(result[2][0].tolist())
        # y_pred_3.append(result[3][0].tolist())
    y_pred = [y_pred_1, y_pred_2, y_pred_3]

    # Create four figures for each of the four param
    fig=plt.figure(1, figsize=(20,12), dpi=300)
    plt.axis("off")

    # r2_list = []

    count_pos = 1
    for pred in y_pred: # for 3 cases
        param_true, param_pred = util.sort_by_param(y_true, pred)
        # r2_by_param = util.rfr_r2_score(y_true, pred)[1]
        # using Spearman rho instead of Pearson's coefficient
        rho_by_param = stats.spearmanr(y_true, pred)
        # for nu1 and nu2 might need to convert y_true, pred back into log scale
        # for more similar r2 than what we have seen before
        # r2_list.append(r2_by_param)

        for i in range(4):
            plt.figure(1)
            fig.add_subplot(3, 4, count_pos)
            # util.plot_by_param(param_true[i], param_pred[i], r2_by_param[i])
            util.plot_by_param(param_true[i], param_pred[i], rho_by_param[0][i][i+4])
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
    plt.ylabel('RFR_1')
    plt.subplot(3, 4, 9)
    plt.ylabel('RFR_avg')

    # plt.show()

    fig.savefig('results/2d-splitmig/benchmarking/benchmarking_dadi_v_RFR_troubleshoot.png', bbox_inches='tight')

    #_no_perturb