import os
import sys
import time
import dadi
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # load testing data set
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-fixed-m','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-vary-T','rb'))
    # list_test_dict = pickle.load(open('data/2d-splitmig/test-data-vary-T-100','rb'))
    list_test_dict = pickle.load(open('data/2d-splitmig/test-data-corrected','rb'))
    # load list of trained rfr
    list_rfr = pickle.load(open('data/2d-splitmig/list-rfr','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-sampling','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-fixed-m','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-10','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-10-sampling','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-finer-Tm','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-100','rb'))
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-vary-T-100-sampling','rb'))

    # Create four figures for each of the four param
    fig1=plt.figure(1, figsize=(22,16), dpi=300)
    # plt.title("nu1")
    # plt.title("nu1-sampling")
    # plt.title("nu1-fixed-m")
    # plt.title("nu1-vary-T")
    # plt.title("nu1-vary-T-10")
    # plt.title("nu1-vary-T-10-sampling")
    # plt.title("nu1-finer-Tm")
    # plt.title("nu1-vary-T-100")
    # plt.title("nu1-vary-T-100-sampling")
    plt.title("nu1-corrected_m")
    plt.axis("off")
    
    fig2=plt.figure(2, figsize=(22,16), dpi=300)
    # plt.title("nu2")
    # plt.title("nu2-sampling")
    # plt.title("nu2-fixed-m")
    # plt.title("nu2-vary-T")
    # plt.title("nu2-vary-T-10")
    # plt.title("nu2-vary-T-10-sampling")
    # plt.title("nu2-finer-Tm")
    # plt.title("nu2-vary-T-100")
    # plt.title("nu2-vary-T-100-sampling")
    plt.title("nu2-corrected_m")
    plt.axis("off")

    fig3=plt.figure(3, figsize=(22,16), dpi=300)
    # plt.title("T")
    # plt.title("T-sampling")
    # plt.title("T-fixed-m")
    # plt.title("T-vary-T")
    # plt.title("T-vary-T-10")
    # plt.title("T-vary-T-10-sampling")
    # plt.title("T-finer-Tm")
    # plt.title("T-vary-T-100")
    # plt.title("T-vary-T-100-sampling")
    plt.title("T-corrected_m")
    plt.axis("off")

    fig4=plt.figure(4, figsize=(22,16), dpi=300)
    # plt.title("m")
    # plt.title("m-sampling")
    # plt.title("m-fixed-m")
    # plt.title("m-vary-T")
    # plt.title("m-vary-T-10")
    # plt.title("m-vary-T-10-sampling")
    # plt.title("m-finer-Tm")
    # plt.title("m-vary-T-100")
    # plt.title("m-vary-T-100-sampling")
    plt.title("m-corrected_m")
    plt.axis("off")

    # logs = [True, True, False, False]

    # testing, and plotting
    count_pos = 1
    for rfr in list_rfr:
        for test_dict in list_test_dict:
            y_true, y_pred = util.rfr_test(rfr, test_dict)
            # sort test results by param for plotting
            param_true, param_pred = util.sort_by_param(y_true, y_pred)
            # calculate r2 and msle scores by param
            r2_by_param = util.rfr_r2_score(y_true, y_pred)[1]
            # msle_by_param = util.rfr_msle(y_true, y_pred)[1]
            
            # PLOT MULTIPLE SUBPLOT VERSION
            plt.figure(1)
            fig1.add_subplot(4, 4, count_pos)
            # util.plot_by_param(param_true[0], param_pred[0], 
            #                 r2_by_param[0], msle_by_param[0])
            log_nu1_true = [10**param_true for param_true in param_true[0]]
            log_nu1_pred =  [10**param_pred for param_pred in param_pred[0]]
            util.plot_by_param(log_nu1_true, log_nu1_pred, r2_by_param[0])

            plt.figure(2)
            fig2.add_subplot(4, 4, count_pos)
            log_nu2_true = [10**param_true for param_true in param_true[1]]
            log_nu2_pred =  [10**param_pred for param_pred in param_pred[1]]
            util.plot_by_param(log_nu2_true, log_nu2_pred, r2_by_param[1])

            plt.figure(3)
            fig3.add_subplot(4, 4, count_pos)
            util.plot_by_param(param_true[2], param_pred[2], 
                            r2_by_param[2])
            
            plt.figure(4)
            fig4.add_subplot(4, 4, count_pos)
            util.plot_by_param(param_true[3], param_pred[3], 
                            r2_by_param[3])
            count_pos += 1

    # # Plot T/m
    # fig1=plt.figure(1, figsize=(22,16), dpi=300)
    # plt.title("T/m")
    # plt.axis("off")

    # # testing, and plotting
    # count_pos = 1
    # for rfr in list_rfr:
    #     for test_dict in list_test_dict:
    #         y_true, y_pred = util.rfr_test(rfr, test_dict)
    #         # sort test results by param for plotting
    #         param_true, param_pred = util.sort_by_param(y_true, y_pred)
    #         # plot true vs pred T/m 
    #         plt.figure(1)
    #         fig1.add_subplot(4, 4, count_pos)
    #         T_over_m_true = [T_true/m_true for T_true, m_true in 
    #                                 zip(param_true[2], param_true[3])]
    #         T_over_m_pred = [T_pred/m_pred for T_pred, m_pred in 
    #                                 zip(param_pred[2], param_pred[3])]
    #         # util.plot_by_param(T_over_m_true, T_over_m_pred, c=param_true[3])
    #         util.plot_by_param(T_over_m_true, T_over_m_pred)
    #         count_pos += 1
    # fig1.savefig('results/2d-splitmig/T_over_m_3.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/nu1.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/nu2.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/T.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/m.png', bbox_inches='tight')
    
    # fig1.savefig('results/2d-splitmig/nu1-sampling.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/nu2-sampling.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/T-sampling.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/m-sampling.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/nu1-fixed-m.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/nu2-fixed-m.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/T-fixed-m.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/m-fixed-m.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-vary-T.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-vary-T.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-vary-T.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-vary-T.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-vary-T-10.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-vary-T-10.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-vary-T-10.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-vary-T-10.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-vary-T-10-sampling.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-vary-T-10-sampling.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-vary-T-10-sampling.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-vary-T-10-sampling.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-finer-Tm.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-finer-Tm.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-finer-Tm.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-finer-Tm.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-vary-T-100.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-vary-T-100.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-vary-T-100.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-vary-T-100.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-vary-T-100-sampling.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-vary-T-100-sampling.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-vary-T-100-sampling.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-vary-T-100-sampling.png', bbox_inches='tight')

    # fig1.savefig('results/2d-splitmig/test/nu1-vary-T-100-sampling-test-100.png', bbox_inches='tight')
    # fig2.savefig('results/2d-splitmig/test/nu2-vary-T-100-sampling-test-100.png', bbox_inches='tight')
    # fig3.savefig('results/2d-splitmig/test/T-vary-T-100-sampling-test-100.png', bbox_inches='tight')
    # fig4.savefig('results/2d-splitmig/test/m-vary-T-100-sampling-test-100.png', bbox_inches='tight')

    fig1.savefig('results/2d-splitmig/test/corrected_m/nu1-corrected_m.png', bbox_inches='tight')
    fig2.savefig('results/2d-splitmig/test/corrected_m/nu2-corrected_m.png', bbox_inches='tight')
    fig3.savefig('results/2d-splitmig/test/corrected_m/T-corrected_m.png', bbox_inches='tight')
    fig4.savefig('results/2d-splitmig/test/corrected_m/m-corrected_m.png', bbox_inches='tight')