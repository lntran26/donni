import os
import sys
import time
import dadi
import numpy as np
import random
import pickle
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util
from sklearn.metrics import mean_squared_log_error, r2_score

if __name__ == '__main__': 
    # open a text file to record experiment results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    sys.stdout = open('results/2d-splitmig/2d-splitmig-'+ timestr +'.txt', 'w')
    # print header to visually seperate each run
    print('*'*70, '\n')
    # print the date and time of run
    print('EXPERIMENT DATE: ', time.asctime(time.localtime(time.time())))
    # print guide key to intepret the numbers for training and testing cases
    print(
    '''
    Keys for Training/Testing #:
    # 1 : no noise
    # 2 : theta = 100
    # 3 : theta = 1,000 
    # 4 : theta = 10,000
    '''
        )

    # load list of training data
    list_train_dict = pickle.load(open('data/2d-splitmig/train-data','rb'))
    # print training set info 
    train_params = list_train_dict[0].keys()
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))
    
    # load list of testing data
    list_test_dict = pickle.load(open('data/2d-splitmig/test-data','rb'))
    test_params = list_test_dict[0].keys()
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))

    # print theta list
    theta_list = [1, 100, 1000, 10000]
    print('Theta list:', theta_list)

    # # load list of trained rfr
    # list_rfr = pickle.load(open('data/2d-splitmig/list-rfr','rb'))

    # load list of trained rfr with sampling
    list_rfr = pickle.load(open('data/2d-splitmig/list-rfr-sampling','rb'))
    # print relevant info for this 
    print('This run uses train set with fs sampling.')

    # Initialize an empty list to store multiple score lists from each training set and replicate
    score_list = []
    
    # Print the R2 scores for all parameters and test case
    # Use count to store key# for each run
    count_train = 1
    for rfr in list_rfr:
        print('\n', 'TRAINING SET #', str(count_train),'\n')
        count_test = 1
        for test_dict in list_test_dict:
            print('TEST CASE # ', str(count_test))
            y_true, y_pred = util.rfr_test(rfr, test_dict)
            score = r2_score(y_true, y_pred)
            score_list.append(score)
            # print('MSLE for each predicted param:', 
            #         mean_squared_log_error(y_true, y_pred, 
            #             multioutput='raw_values'))
            # print('Aggr. MSLE for all predicted params:', mean_squared_log_error(y_true, y_pred),'\n')
            print('R2 score for each predicted param:', 
                        r2_score(y_true, y_pred, multioutput='raw_values'))
            print('Aggr. R2 score for all predicted params:', 
                        r2_score(y_true, y_pred),'\n')
            count_test += 1
        count_train += 1
    
    print('END OF RUN', '\n')
    # close the text file
    sys.stdout.close()
    # exit mode
    sys.exit()