### DATE: 09/12/2020
### LINH TRAN

### This program provide functions to make training and testing data for 
### the RFR algorithm from a user-provide list of parameters, and to train and ### test the RFR algorithm on those data sets.
### Output is saved into a text file including the inferred parameter values
### and the avarage R2 scores to determine how the algorithm performs
### with different training and testing data sets.

### New in this version: ADD TO RFR_TRAIN TO RETURN R2SCORE LIST
### User guide: user may want to customize: 
### theta_list, number of experimental replicates,
### parameter values for testing and training, and output file name

import dadi
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random

import os
import sys
import time

# Function: Used to generate a list of dictionaries of params and FS for
# training and testing purposes.
# Dependency: dadi, theta list
# Input: a list of parameters from which to generate fs; under the
# theta=1, 100(+), 1000, and 10000 conditions, and a list specifying theta
# NOTES: theta=100 can sometimes lead to NaN errors so I increased it to 100+
# Output: return a list of four dictionaries with theta unscaled or scaled, 
# each dictionary has structure params:fs
def make_list_dicts(params_list, theta_list):
    # one pop, two epoch model 
    func = dadi.Demographics1D.two_epoch
    func_ex = dadi.Numerics.make_extrap_func(func)
    # specify sample size and extrapolation grid
    ns = [20]
    pts_l = [40, 50, 60]
    # specify list of theta values and initialize output list of dictionaries
    list_theta = theta_list
    list_dicts = []
    # populate fs values into each dictionary of the list of dictionaries
    for i in range(len(list_theta)):
        data_dict = {}
        theta = list_theta[i]
        # generate data sets by looping through list of
        # parameters and generate corresponding fs
        # then one finish the whole list save as a dictionary
        if theta == 1:
            # need to seperate this case since no sampling
            for params in params_list:
                # generate spectrum
                fs = func_ex(params, ns, pts_l)
                # normalize all fs
                fs_norm = fs/fs.sum()
                # store spectrum data in dictionary
                data_dict[params] = fs_norm
        else:
            for params in params_list:
                # generate spectrum
                fs = func_ex(params, ns, pts_l)
                # scale by theta and randomly sample
                fs_scaled = (theta*fs).sample()
                # normalize all fs
                fs_scaled_norm = fs_scaled/fs_scaled.sum() 
                # store spectrum data in dictionary
                data_dict[params] = fs_scaled_norm
        # store dictionary generated to dictionary list
        list_dicts.append(data_dict)
    return list_dicts

# Function: Train the random forest regressor algorithm on a particular
# training set, then test with a variety of testing sets.
# Dependency: Scikit learn library
# Input: one dictionary of training set to train & a list of testing
# data sets (dictionaries) to test on
# Output: print R2Score fit on training data, prediction values, 
# and R2Score for testing data sets
# Return output: test R2 scores stored in a list of lists
def rfr_train(train_dict, list_test_dict):
    X, y = [], []
    for params in train_dict:
        y.append(params)
        X.append(train_dict[params].data)
    # Fit regression model
    rfr = RandomForestRegressor()
    rfr = rfr.fit(X, y)
    print('R2 score with train data: ', rfr.score(X, y), '\n')

    score_list = []
    count = 1 # Use count to print key# for each run
    for test_dict in list_test_dict:
        print('TEST CASE # ', str(count))
        X_test, y_test = [], []
        for params in test_dict:
            input = test_dict[params].data
            print('Expected params: ', str(params), 
                ' vs. Predict params: ', str(rfr.predict([input])))
            y_test.append(params)
            X_test.append(test_dict[params].data)
        print('R2 score with test data: ', rfr.score(X_test, y_test), '\n')
        score_list.append(rfr.score(X_test, y_test))
        count += 1
    return score_list

# generate parameter list for training
train_params = []
# step arg (3rd) in np.arrange determine how dense or sparse the training data
# change nu from 10e-2 to 10e2, increment e by 0.3
for nu in [10**i for i in np.arange(-2, 2.1, 0.3)]:
    # change T from 0.1 to 2, increment by 0.1
    for T in np.arange(0.1, 2.1, 0.1):
        # params tuple for this spectrum
        params = (round(nu, 2), round(T, 1))
        train_params.append(params)
# print('n_samples training: ', len(train_params))

# generate parameter list for testing
test_params = []
# range(#) dictate how many values are in each test set
for i in range(50):
# generate random nu and T within the same range as training data range
    nu = 10 ** (random.random() * 4 - 2)
    T = random.random() * 1.9 + 0.1
    params = (round(nu, ndigits=2), round(T, ndigits=1))
    test_params.append(params)
# print('n_samples testing: ', len(test_params))

# choose list of theta values to run scaling and add variance
theta_list = [1, 150, 1000, 10000]

# Use function to make lists of dictionaries storing different training and testing data sets from lists of parameters
list_train_dict = make_list_dicts(train_params, theta_list)
# print('Number of train dicts: ', len(list_train_dict))
# print('n_samples in train dicts: ', len(list_train_dict[3]))

list_test_dict = make_list_dicts(test_params, theta_list)
# print('Number of test dicts: ', len(list_test_dict))
# print('n_samples in test dicts: ', len(list_test_dict[0]))

# open a text file to save the print output
sys.stdout = open("1d-2epoch.txt", "a")
# print the date and time of run
print('Experiment date: ', time.asctime(time.localtime(time.time())))
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
print('Theta = 100 in this run is replaced by theta =', str(theta_list[1]))

# Use for loop to run several replicates decided by range(#)
R2_score_list = []
# Assign number of replicates to be run
num_rep = 3
print('Number of replicates in this run:', num_rep)
# Use for loop to repeat the number of replicates designated
for i in range(num_rep):
    count = 1 # Use count to store key# for each run
    # Use for loop to train with different training dicts in list;
    # print out the inferred parameters for each training set and
    # return the test R2 scores in a list
    for train_dict in list_train_dict:
        print('TRAINING SET # ', str(count))
        R2_score_list.append(rfr_train(train_dict, list_test_dict))
        count += 1
# Calculate and print out average scores of all replicate runs:
# loop over the original score list and pick out the sets of score
# for each training set that corresponds to different replicates
# of the same training set then avarage the scores from all reps
for i in range(len(list_train_dict)):
    rep_score_list = [R2_score_list[i]]
    while len(rep_score_list) < num_rep:
        # len(list_train_dict) specifies the number of train dict
        # in each replicate so it guides how many [] to jump to
        rep_score_list.append(R2_score_list[i+len(list_train_dict)])
    print('Raw test scores list for TRAINING set #:', i+1,':', rep_score_list)

    # calculate the mean score of all test cases for that training set
    # by converting each list into a np array and calculate avarge
    mean_scores = np.mean(np.array(rep_score_list), axis=0)
    # mean_scores is a list of scores for 1 training set averaged
    # over all replicates for that training set
    # print out results
    # print('Experimental replicates #: ', num_rep)
    print ('Average test scores for TRAINING set #', i+1,':')
    for j in range(len(mean_scores)):
        print('\t','Average scores for test case #', j+1,':', 
            round(mean_scores[j], 2),'\n')
print('\n\n')
# close the text file
sys.stdout.close()
# exit mode
sys.exit()