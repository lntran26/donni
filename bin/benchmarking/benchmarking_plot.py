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

    result_set = pickle.load(open('bin/benchmarking/benchmarking_results_set','rb'))
    # result_set_dup = pickle.load(open('benchmarking_results_set_rerun','rb'))
    # result_set_2 = pickle.load(open('benchmarking_results_set_2','rb'))

    # result_set # length 20
    # result_set_dup[0]
    # type(result_set_dup[0][1])
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
    fig1=plt.figure(1, figsize=(12,6), dpi=300)
    plt.title("nu1-benchmarking")
    plt.axis("off")
    
    fig2=plt.figure(2, figsize=(12,6), dpi=300)
    plt.title("nu2-benchmarking")
    plt.axis("off")

    fig3=plt.figure(3, figsize=(12,6), dpi=300)
    plt.title("T-benchmarking")
    plt.axis("off")

    fig4=plt.figure(4, figsize=(12,6), dpi=300)
    plt.title("m-benchmarking")
    plt.axis("off")

    count_pos = 1
    for pred in y_pred: # for 3 cases
        param_true, param_pred = util.sort_by_param(y_true, pred)
        r2_by_param = util.rfr_r2_score(y_true, pred)[1]
        
        plt.figure(1)
        fig1.add_subplot(1, 3, count_pos)
        # nu1
        util.plot_by_param(param_true[0], param_pred[0], r2_by_param[0])

        plt.figure(2)
        fig2.add_subplot(1, 3, count_pos)
        # nu2
        util.plot_by_param(param_true[1], param_pred[1], r2_by_param[1])

        plt.figure(3)
        fig3.add_subplot(1, 3, count_pos)
        util.plot_by_param(param_true[2], param_pred[2], r2_by_param[2])
        
        plt.figure(4)
        fig4.add_subplot(1, 3, count_pos)
        util.plot_by_param(param_true[3], param_pred[3], r2_by_param[3])
        
        count_pos += 1

    # plt.show()

    fig1.savefig('results/2d-splitmig/benchmarking/nu1-benchmarking.png', bbox_inches='tight')
    fig2.savefig('results/2d-splitmig/benchmarking/nu2-benchmarking.png', bbox_inches='tight')
    fig3.savefig('results/2d-splitmig/benchmarking/T-benchmarking.png', bbox_inches='tight')
    fig4.savefig('results/2d-splitmig/benchmarking/m-benchmarking.png', bbox_inches='tight')
