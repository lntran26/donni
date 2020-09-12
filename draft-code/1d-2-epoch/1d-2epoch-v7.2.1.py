### DATE: 09/11/2020
### LINH TRAN

### This program train the RFR algorithm with the more dense
### input data by reducing the sparseness in the training set.
### Then we test with a more randomized training data.

### New in this version:
### Cleaning up the code with loops and function to shorten the code 
### and avoid repetitive scripts.
### Provides comprehensive pairwise comparison to train and test
### on both no noise and noisy data.

import dadi
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random

import os
import sys
import time

# Function: Used to generate a list of dictionaries of params and FS for
# training and testing purposes.
# Dependency: dadi
# Input: a list of parameters from which to generate fs; under the
# theta=1, 100(+), 1000, and 10000 conditions
# NOTES: theta=100 can sometimes lead to NaN errors so I increased it to 100+
# Output: return a list of four dictionaries with theta unscaled or scaled, 
# each dictionary has structure params:fs

# choose list of theta values to run
theta_list = [1, 150, 1000, 10000]

def make_list_dicts(params_list):
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
def rfr_train(train_dict, list_test_dict):
    X, y = [], []
    for params in train_dict:
        y.append(params)
        X.append(train_dict[params].data)
    # Fit regression model
    rfr = RandomForestRegressor()
    rfr = rfr.fit(X, y)
    print('R2 score with train data: ', rfr.score(X, y), '\n')

    count = 1
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
        count += 1

# generate parameter list for training
train_params = []
for nu in [10**i for i in np.arange(-2, 2.1, 0.3)]:
    # change T from 0.1 to 2, increment by 0.1
    for T in np.arange(0.1, 2.1, 0.1):
        # params tuple for this spectrum
        params = (round(nu, 2), round(T, 1))
        train_params.append(params)
# print('n_samples training: ', len(train_params))

# generate parameter list for testing
test_params = []
for i in range(50):
# generate random nu and T
    nu = 10 ** (random.random() * 4 - 2)
    T = random.random() * 1.9 + 0.1
    params = (round(nu, ndigits=2), round(T, ndigits=1))
    test_params.append(params)
# print('n_samples testing: ', len(test_params))

# initialize a list of dictionaries storing different training and testing data sets
list_train_dict = make_list_dicts(train_params)
# print('Number of train dicts: ', len(list_train_dict))
# print('n_samples in train dicts: ', len(list_train_dict[3]))

list_test_dict = make_list_dicts(test_params)
# print('Number of test dicts: ', len(list_test_dict))
# print('n_samples in test dicts: ', len(list_test_dict[0]))

# open a text file to save the print output
sys.stdout = open("1d-2epoch-noise.txt", "a")
# print the date and time of run
print('Experiment date: ', time.asctime(time.localtime(time.time())))
print(
'''
Keys for Training/Testing #:
# 1 : no noise
# 2 : theta = 100+
# 3 : theta = 1,000 
# 4 : theta = 10,000
'''
    )
print('Theta=100 in this run is replaced by theta =', str(theta_list[1]))
# Use for loop to train with different training lists
# and print out the results for each training set
count = 1
for train_dict in list_train_dict:
    print('TRAINING SET # ', str(count))
    rfr_train(train_dict, list_test_dict)
    count += 1
print('\n\n')
# close the text file
sys.stdout.close()
# exit mode
sys.exit()