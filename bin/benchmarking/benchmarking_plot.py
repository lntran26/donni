import pickle
import dadi
import numpy as np
import util
import matplotlib.pyplot as plt

if __name__ == '__main__': 
    # import result files from hpc and test files
    # test files are just to check to make sure results are expected
    # test_set = pickle.load(open('benchmarking_test_set','rb'))
    # test_set_2 = pickle.load(open('benchmarking_test_set_2','rb'))
    # test_set_3 = pickle.load(open('benchmarking_test_set_3','rb'))

    # result_set = pickle.load(open(
    #   'bin/benchmarking/benchmarking_results_set','rb'))
    result_set = pickle.load(open(
        'bin/benchmarking/benchmarking_results_set_rerun','rb'))
    # result_set = pickle.load(open(
    #     'bin/benchmarking/benchmarking_results_set_2','rb'))
    # result_set = pickle.load(open(
    #   'bin/benchmarking/benchmarking_results_set_3','rb'))
    # result_set = pickle.load(open(
    #     'bin/benchmarking/benchmarking_results_set_no_perturb_2','rb'))
    # result_set = pickle.load(open(
    #     'bin/benchmarking/benchmarking_results_set_no_perturb_3','rb'))

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
    plt.title("benchmarking test set 1")
    plt.axis("off")

    # r2_list = []

    count_pos = 1
    for pred in y_pred: # for 3 cases
        param_true, param_pred = util.sort_by_param(y_true, pred)
        r2_by_param = util.rfr_r2_score(y_true, pred)[1]
        # r2_list.append(r2_by_param)

        for i in range(4):
            plt.figure(1)
            fig.add_subplot(3, 4, count_pos)
            util.plot_by_param(param_true[i], param_pred[i], r2_by_param[i])
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

    # plt.show()

    fig.savefig('results/2d-splitmig/benchmarking/benchmarking_dup.png', bbox_inches='tight')

    #_no_perturb