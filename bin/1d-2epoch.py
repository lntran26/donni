### DATE UPDATED: 09/12/2020
### LINH TRAN, CONNIE SUN

### This program provide functions to make training and testing data for 
### the RFR algorithm with the 1D, 2Epoch model, from a user-provide list of
### parameters, and to train and ### test the RFR algorithm on those data sets.
### Output is saved into a text file including the inferred parameter values
### and the avarage R2 scores to determine how the algorithm performs
### with different training and testing data sets.

### User guide: user may want to customize: 
### List of theta values: theta_list, 
### Number of experimental replicates: num_rep,
### Parameter values for testing and training: train_params and test_params   
### and local output directory and file name: path/to/file/output.txt

import os
import sys
import time

import dadi
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random

def make_list_dicts(params_list, theta_list):
    '''Returns a list of dictionaries, each dictionary in the list has structure 
    params:fs with expected fs for the 1D-2epoch model and corresponds with each 
    theta value specified by user. 
    Useful for creating many variations of training and testing data sets from 
    the same set of parameter values by scaling with different thetas.
    Input: a list of parameters from which to generate fs; a list specifying theta values.
    NOTES: theta =! 1 && <=150 can sometimes lead to NaN errors so recommend increasing 
    theta=100 to 150+ (>=200 works so far)
    Require library: dadi
    '''
    # use the 1D-2epoch model function from dadi
    func = dadi.Demographics1D.two_epoch
    # make the extrapolated version of our demographic model function.
    func_ex = dadi.Numerics.make_extrap_func(func)
    # specify sample size and extrapolation grid
    ns = [20]
    pts_l = [40, 50, 60]
    # specify list of theta values and initialize output list of dictionaries
    list_dicts = []
    # populate fs values into each dictionary of the list of dictionaries
    
    for theta in theta_list:
        data_dict = {}
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

def rfr_train(train_dict, list_test_dict):
    '''Trains a RandomForestRegressor algorithm and tests its performance. 
    Returns a list of R2 scores from testing the specified training data set with 
    the specified list of testing data sets. Also print R2 score fit for the 
    training data, all inferred parameter values for each testing data set, 
    and R2 score for that test set.
    The R2 scores list can be used to calculate average scores when running 
    multiple replicate experiments on the same training and testing conditions.
    Input: one dictionary of training set; a list of testing set dictionaries
    Require library: Scikit learn ensemble random forest regressor
    '''
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

# open a text file to record experiment results
timestr = time.strftime("%Y%m%d-%H%M%S")
sys.stdout = open('results/1d-2epoch/1d-2epoch-'+ timestr +'.txt', 'w')
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
train_params = []
# step arg (3rd) in np.arrange determine how dense or sparse the training data
# change nu from 10e-2 to 10e2, increment e by 0.2
for nu in [10**i for i in np.arange(-2, 2.1, 0.2)]:
    # change T from 0.1 to 2, increment by 0.1
    for T in np.arange(0.1, 2.1, 0.1):
        # params tuple for this spectrum
        params = (round(nu, 2), round(T, 1))
        train_params.append(params)
print('n_samples training: ', len(train_params))
print('Range of training params:', min(train_params), 'to', 
        max(train_params))

# generate parameter list for testing
test_params = []
# range(#) dictate how many random values are in each test set
for i in range(100):
# generate random nu and T within the same range as training data range
    nu = 10 ** (random.random() * 4 - 2)
    T = random.random() * 1.9 + 0.1
    params = (round(nu, ndigits=2), round(T, ndigits=1))
    test_params.append(params)
print('n_samples testing: ', len(test_params))
print('Range of testing params:', min(test_params), 'to', 
        max(test_params))

# choose list of theta values to run scaling and add variance
theta_list = [1, 200, 1000, 10000]
print('Theta list:', theta_list)

# Use function to make lists of dictionaries storing different training 
# and testing data sets from lists of parameters
list_train_dict = make_list_dicts(train_params, theta_list)
list_test_dict = make_list_dicts(test_params, theta_list)

# Assign number of replicates to be run
num_rep = 3
print('Number of replicates in this run:', num_rep, '\n')

# Initialize an empty list to store multiple score lists from each training set 
# and replicate
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
        R2_score_list.append(rfr_train(train_dict, list_test_dict))
        count += 1

# Calculate and print the average scores of all replicate runs
# by looping over the R2 score list of lists, grouping all the small lists
# that correspond to different replicates of the same training set
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