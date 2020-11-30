import os
import sys
import time
import dadi
import numpy as np
import random
import matplotlib.pyplot as plt
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # generate parameter list for training (nu full range)
    # exclude params where T/nu > 5
    train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
                          for T in np.linspace(0.1, 2, 20) if T/nu <= 5]

    # train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
    #                       for T in np.linspace(0.1, 2, 20)]

    # train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
    #                       for T in np.linspace(0.05, 2, 20) if T/nu <= 5]

    # train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
    #                       for T in np.linspace(0.01, 2, 20) if T/nu <= 5]

    # try smaller range of nu:
    # train_params = [(nu,T) for nu in 10**np.linspace(-2, 0, 21)
    #                     for T in np.linspace(0.1, 2, 20)]
    # nu large range:
    # train_params = [(nu,T) for nu in 10**np.linspace(0, 2, 21)
    #                     for T in np.linspace(0.1, 2, 20)]

    # train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
    #                     for T in np.linspace(0.1, 2, 20) if T/nu <= 50]

    # print training set info 
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))

    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test set
    for i in range(150):
    # generate random nu and T within the same range as training data range
        nu = 10 ** (random.random() * 4 - 2)
        # nu = 10 ** (random.random() * -2)
        # nu = 10 ** (random.random() * 2)
        T = random.random() * 1.9 + 0.1
        # T = random.random() * 1.95 + 0.05
        # T = random.random() * 1.99 + 0.01
        if T/nu <= 5:
            params = (nu, T)
            test_params.append(params)
        # params = (nu, T)
        # test_params.append(params)

    # print testing set info 
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))

    # generate a list of theta values to run scaling and add variance
    ### TO DO: CAN SEPARATE TRAIN AND TEST THETA LISTS
    theta_list = [1,100,1000,10000]
    #print('Theta list:', theta_list)
    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # Use function to make lists of dictionaries storing different training and 
    # testing data sets from lists of parameters
    list_train_dict = util.generating_data_parallel(train_params, 
                        theta_list, func, ns, pts_l)
    list_test_dict = util.generating_data_parallel(test_params, 
                        theta_list, func, ns, pts_l)

    # LOG TRANSFORM VERSION
    # VERSION 1 - DO UNLOG TRANSFORM ON TEST DICT
    # log transform training data for nu
    transformed_list_train_dict = util.log_transform_data(list_train_dict, [0])
    
    # Create two figures one for nu and one for T
    fig1=plt.figure(1, figsize=(22,16), dpi=300)
    # fig1, axes1 = plt.subplots(nrows=4, ncols=4, num=1, figsize=(26,16), squeeze=False)
    # for ax, col in zip(axes1[0], cols):
    #     ax.set_title(col)
    #     ax.title.set_size(14)
    #     # ax.axis("off")
    # for ax, row in zip(axes1[:,0], rows):
    #     ax.set_ylabel(row, rotation='vertical')
    #     ax.yaxis.label.set_size(14)
    plt.title("nu")
    plt.axis("off")
    
    fig2=plt.figure(2, figsize=(22,16), dpi=300)
    # fig2, axes2 = plt.subplots(nrows=4, ncols=4, num=2, figsize=(26,16), squeeze=False)
    # for ax, col in zip(axes2[0], cols):
    #     ax.set_title(col)
    #     ax.title.set_size(12)
    #     ax.axis("off")
    # for ax, row in zip(axes2[:,0], rows):
    #     ax.set_ylabel(row, rotation='vertical')
    #     ax.yaxis.label.set_size(12)
    #     ax.axis("off")
    plt.title("T")
    plt.axis("off")

    # plt.xlabel("Test theta")
    # plt.ylabel("Train theta")
    # plt.xticks([])
    # plt.yticks([])

    # training, testing, and plotting
    count_train = 1
    count_pos = 1
    for train_dict in transformed_list_train_dict:
    # for train_dict in list_train_dict:
        print(count_train)
        rfr = util.rfr_train(train_dict)
        count_test = 1
        # for test_dict in transformed_list_test_dict:
        for test_dict in list_test_dict:
            y_true, y_predict = util.rfr_test(rfr, test_dict)
            # reconvert log for y_predict nu
            new_y_predict = util.un_log_transform_predict(y_predict, [0])
            # sort results of ML prediction by param                               
            param_true, param_pred = util.sort_by_param(y_true, new_y_predict)
            # make list of T/nu based on param_true values
            T_over_nu = [T/nu for T, nu in zip(param_true[1], param_true[0])]
            # calculate r2 and msle scores
            r2_by_param = util.rfr_r2_score(y_true, new_y_predict)[1]
            msle_by_param = util.rfr_msle(y_true, new_y_predict)[1]
            
            # PLOT MULTIPLE SUBPLOT VERSION
            plt.figure(1)
            fig1.add_subplot(4, 4, count_pos)
            util.plot_by_param(param_true[0], param_pred[0], 
                            r2_by_param[0], msle_by_param[0], T_over_nu)

            plt.figure(2)
            fig2.add_subplot(4, 4, count_pos)
            util.plot_by_param(param_true[1], param_pred[1], 
                            r2_by_param[1], msle_by_param[1], T_over_nu)

            # PLOT AND SAVE INDIVIDUAL PLOT VERSION
            # count_param = 1
            # for true, pred, r2, msle in zip(param_true, param_pred, 
            # r2_by_param, msle_by_param):
            # # for true, pred in zip(param_true, param_pred):
            #     util.plot_by_param(true, pred, r2, msle, T_over_nu)
            #     # util.plot_by_param(true, pred)
            #     plt.savefig('train'+str(count_train)+'test'+str(count_test)+
            #     'param'+str(count_param)+'.png', bbox_inches='tight')
            #     plt.clf()
            #     print("count_param is: ", count_param)
            #     count_param+=1
            count_test+=1
            count_pos+=1
        count_train+=1
    # plt.show()
    cols = ['Test theta {}'.format(col) for col in theta_list]
    rows = ['Train theta {}'.format(row) for row in theta_list]
    # plt.figure(1)
    # print(type(fig1.axes))
    # print(len(fig1.axes))
    # print(fig1.axes)
    # print(type(fig1.axes[0]))
    # print(fig1.axes[0])
    # for ax, col in zip(fig1.axes[0], cols):
    #     ax.set_title(col)
    #     ax.title.set_size(14)
    # for ax, row in zip(fig1.axes[:,0], rows):
    #     ax.set_ylabel(row, rotation='vertical')
    #     ax.yaxis.label.set_size(14)
    
    # plt.figure(2)
    # for ax, col in zip(fig2.axes[0], cols):
    #     ax.set_title(col)
    #     ax.title.set_size(14)
    # for ax, row in zip(fig2.axes[:,0], rows):
    #     ax.set_ylabel(row, rotation='vertical')
    #     ax.yaxis.label.set_size(14)

    # fig1.savefig('nu.png', bbox_inches='tight')
    # fig2.savefig('T.png', bbox_inches='tight')

    fig1.savefig('nu_exclude.png', bbox_inches='tight')
    fig2.savefig('T_exclude.png', bbox_inches='tight')

    # # VERSION 2 - DO NOT USE UNLOG TRANSFORM
    # # log transform test dict version
    # transformed_list_train_dict = util.log_transform_data(list_train_dict, [0])
    # transformed_list_test_dict = util.log_transform_data(list_test_dict, [0])
    # # training, testing, and plotting
    # count_train = 1
    # for train_dict in transformed_list_train_dict:
    #     print(count_train)
    #     rfr = util.rfr_train(train_dict)
    #     count_test = 1
    #     for test_dict in transformed_list_test_dict:
    #         y_true, y_predict = util.rfr_test(rfr, test_dict)
    #         param_true, param_pred = util.sort_by_param(y_true, y_predict)
    #         count_param = 1
    #         for true, pred in zip(param_true, param_pred):
    #             util.plot_by_param(true, pred)
    #             plt.savefig('train'+str(count_train)+'test'+str(count_test)+
    #             'param'+str(count_param)+'.png')
    #             plt.clf()
    #             count_param+=1
    #         count_test+=1
    #     count_train+=1

    # # NO LOG TRANSFORM VERSION
    # # training, testing, and plotting
    # count_train = 1
    # for train_dict in list_train_dict:
    #     print(count_train)
    #     rfr = util.rfr_train(train_dict)
    #     count_test = 1
    #     for test_dict in list_test_dict:
    #         y_true, y_predict = util.rfr_test(rfr, test_dict)
    #         param_true, param_pred = util.sort_by_param(y_true, y_predict)
    #         r2_by_param = util.rfr_r2_score(y_true, y_predict)[1]
    #         msle_by_param = util.rfr_msle(y_true, y_predict)[1]
    #         count_param = 1
    #         for true, pred, r2, msle in zip(param_true, param_pred, 
    #         r2_by_param, msle_by_param):
    #             util.plot_by_param(true, pred, r2, msle)
    #             plt.savefig('train'+str(count_train)+'test'+str(count_test)+
    #             'param'+str(count_param)+'.png')
    #             plt.clf()
    #             count_param+=1
    #         count_test+=1
    #     count_train+=1
