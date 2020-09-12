#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:55:02 2020

@author: conniesun

This file tests with a nu that changes in log scale. 
Tests with non-noisy data that falls completely within
the training data params.


training data 
    nu: goes from 0.01 to 100, changes in intervals of 10^i, where
        i changes from -2 to 2 with step size of 0.5
    T: goes from 0.5 to 2 with step size of 0.5
    
testing data 
    nu: goes from 0.01 to 100 changes in intervals of 10^i, where
        i changes from -2 to 2 with step size of 0.2
    T: goes from 0.6 to 2 with step size of 0.2
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 09:52:06 2020

@author: conniesun
"""
import dadi
import numpy as np

# 1 population, two epoch model 
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_func(func)

# unchanged
ns = [20]
pts_l = [40, 50, 60]

# construct training dictionary
train_dict = {}
# change nu from 0.01 to 100, log scale 
for nu in [10**i for i in np.arange(-2, 2.1, 0.5)]:
    # change T from 0.5 to 2, increment by 0.5
    for T in np.arange(0.5, 2.1, 0.5):
        #params list for this spectrum
        params = (round(nu, 2), T)
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        # fsnorm = fs/fs.sum() # normalize all fs
        train_dict[params] = fs
        
# construct testing dictionary
test_dict = {}
# change nu from 0.01 to 100, log scale
for nu in [10**i for i in np.arange(-2, 2.1, 0.2)]:
    # change T from 0.6 to 2.0, increment by .2
    for T in np.arange(0.6, 2.1, 0.2):
        params = (round(nu, 2), round(T, 1))
        fs = func_ex(params, ns, pts_l)
        test_dict[params] = fs

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
print("\nTrain data fit:", rfr.score(X, y))
print("Test data fit:", rfr.score(X_test, y_test), '\n')



