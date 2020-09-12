#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:10:48 2020

@author: conniesun

Random sampling test -- increased training data density
and create random sets of params for testing 
"""

import dadi
import numpy as np
import random

# 1 population, two epoch model 
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_func(func)

# unchanged
ns = [20]
pts_l = [40, 50, 60]

# construct training dictionary
train_dict = {}
# change nu from 0.01 to 100, log scale 
for nu in [10**i for i in np.arange(-2, 2.1, 0.2)]:
    # change T from 0.1 to 2, increment by 0.1
    for T in np.arange(0.1, 2.1, 0.1):
        #params list for this spectrum
        params = (round(nu, 2), T)
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        fsnorm = fs/fs.sum() # normalize all fs
        train_dict[params] = fsnorm
        
# construct testing dictionary
test_dict = {}
# generate 100 random tests within range
for i in range(100):
    # generate random nu and T
    nu = 10 ** (random.random() * 4 - 2)
    T = random.random() * 1.9 + 0.1
    params = (round(nu, ndigits=2), round(T, ndigits=1))
    fs = func_ex(params, ns, pts_l)
    fsnorm = fs/fs.sum() # normalize all fs
    test_dict[params] = fsnorm

# construct RF and train
from sklearn.ensemble import RandomForestRegressor
X, y = [], []
# create training lists from dictionary
for params in train_dict:
    y.append(params)
    X.append(train_dict[params].data)
# fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)

# testing data
X_test, y_test = [], []
for params in test_dict:
    # create the testing lists for r^2 score
    y_test.append(params)
    X_test.append(test_dict[params].data)
    # create inputs and print predictions
    test_input = test_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([test_input])))

# r^2 scores
print("Train data fit:", rfr.score(X, y), '\n')
print("Test data fit:", rfr.score(X_test, y_test), '\n')




