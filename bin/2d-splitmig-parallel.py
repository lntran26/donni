import os
import sys
import time
import dadi
import numpy as np
import random
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util

if __name__ == '__main__': 
    # print("main script is run directly")
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
    # 2 : theta = 100+
    # 3 : theta = 1,000 
    # 4 : theta = 10,000
    '''
        )
    # generate parameter list for training
    train_params = [(nu1, nu2, T, m) for nu1 in 10**np.linspace(-2, 2, 3)
                                    for nu2 in 10**np.linspace(-2, 2, 3)
                                    for T in np.linspace(0.1, 2, 3)
                                    for m in np.linspace(1, 10, 3)]
    print('n_samples training: ', len(train_params))
    print('Range of training params:', min(train_params), 'to', 
            max(train_params))

    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test set
    for i in range(50):
    # generate random nu and T within the same range as training data range
        nu1 = 10 ** (random.random() * 4 - 2)
        nu2 = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        m = random.random() * 9.9 + 0.1
        params = (round(nu1, 2), round(nu2, 2), round(T, 1), round(m, 1))
        test_params.append(params)
    # print(some info of testing data)
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))

    # generate a list of theta values to run scaling and add variance
    theta_list = [1]
    print('Theta list:', theta_list)

    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]

    # Use function to make lists of dictionaries storing different training and 
    # testing data sets from lists of parameters
    list_train_dict = util.generating_data_parallel(train_params, 
                        theta_list, func, ns, pts_l)
    list_test_dict = util.generating_data_parallel(test_params, 
                        theta_list, func, ns, pts_l)

    # Assign number of replicates to be run
    num_rep = 3
    print('Number of replicates in this run:', num_rep, '\n')

    # Initialize an empty list to store multiple score lists from each training set and replicate
    R2_score_list = []

    # Use for loop to run several replicates designated by num_rep,
    # for each rep, use for loop to train with different training dicts 
    # in list of training dicts and test each case against all test dicts.     
    # Print out the inferred parameters and R2 scores for all replicates and
    # store all the test R2 scores in a list of lists, where each small list
    # represents all R2 test scores for one training set in one replicate.
    for i in range(num_rep):
        print('-'*20, 'REPLICATE #', i+1, '-'*20)
        # Use count to store key# for each run
        count = 1
        for train_dict in list_train_dict:
            print('TRAINING SET #', str(count))
            R2_score_list.append(util.rfr_learn(train_dict, list_test_dict))
            count += 1
    # Calculate and print the average scores of all replicate runs
    # by looping over the R2 score list of lists, grouping all the small lists
    # that correspond to different replicates of the same training set,
    # then avarage to get the mean R2 test scores for that training set 
    # across all replicate runs.
    for i in range(len(list_train_dict)):
        # len(list_train_dict) specifies the number of training sets used
        # in each replicate, which dictates the distance between replicates
        # of the same training set in the score list
        rep_score_list = [R2_score_list[i]]
        while len(rep_score_list) < num_rep:
            # len(rep_score_list) should equals num_rep before stopping
            rep_score_list.append(R2_score_list[i+len(list_train_dict)])
        print('Raw test scores list for TRAINING set #', i+1,':', 
            rep_score_list, '\n')
        # calculate the mean score of all test cases for that training set
        # by converting each rep_score_list, which is a list of list
        # into a np array so we can easily calculate the means of the same test
        # for each training set by columns
        mean_scores = np.mean(np.array(rep_score_list), axis=0)
        # mean_scores is a list of test scores for 1 training set against many test sets, 
        # averaged over all replicates for that training set
        # print out results
        print ('Average test scores for TRAINING set #', i+1,':')
        for j in range(len(mean_scores)):
            print('\t','Average scores for test case #', j+1,':', 
                round(mean_scores[j], 2))
        print('\n')
    print('END OF RUN', '\n')
    # close the text file
    sys.stdout.close()
    # exit mode
    sys.exit()
# else:
#     print("main script is not running directly")

