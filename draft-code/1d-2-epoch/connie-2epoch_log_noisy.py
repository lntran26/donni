#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:24:37 2020

@author: conniesun

This file tests with a nu that changes in log scale. 
Tests with noisy data that falls within the same training
params but with theta 100, 1000, and 10000. 


training data 
    nu: goes from 0.01 to 100, changes in intervals of 10^i, where
        i changes from -2 to 2 with step size of 0.5
    T: goes from 0.5 to 2 with step size of 0.5
    
testing data 
    "noisy": theta of 100, 1000, 10000
    nu: goes from 0.01 to 100 changes in intervals of 10^i, where
        i changes from -2 to 2 with step size of 0.2
    T: goes from 0.6 to 2 with step size of 0.2
"""
import pylab
import dadi
import numpy as np

# 1 population, two epoch model 
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_func(func)

# unchanged
ns = [20]
pts_l = [40, 50, 60]

train_dict = {}
#create noisy data for testing       
test_dict100 = {}
test_dict1000 = {}
test_dict10000 = {} 

# change nu from 0.01 to 100, log scale 
for nu in [10**i for i in np.arange(-2, 2.1, 0.5)]:
    T = 0.5
    # change T from 0.5 to 2, increment by 0.5
    for T in np.arange(0.5, 2.1, 0.5):
        #params list for this spectrum
        params = (round(nu, 2), T)
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        fsnorm = fs/fs.sum() # normalize all fs
        train_dict[params] = fsnorm
        
        #theta 100
        fs100 = (100*fs).sample()
        fs100norm = fs100/fs100.sum()
        test_dict100[params] = fs100norm
        
        #theta = 1000
        fs1000 = (1000*fs).sample()
        fs1000norm = fs1000/fs1000.sum()
        test_dict1000[params] = fs1000norm
        
        #theta = 10000
        fs10000 = (10000*fs).sample()
        fs10000norm = fs10000/fs10000.sum()
        test_dict10000[params] = fs10000norm

#plotting the models 
fsnorm = train_dict[(10, 0.5)]
dadi.Plotting.plot_1d_fs(fsnorm)
pylab.show()

fs100norm = test_dict100[(10, 0.5)]
dadi.Plotting.plot_1d_fs(fs100norm)
pylab.show()

fs1000norm = test_dict1000[(10, 0.5)]
dadi.Plotting.plot_1d_fs(fs1000norm)
pylab.show()

fs10000norm = test_dict10000[(10, 0.5)]
dadi.Plotting.plot_1d_fs(fs10000norm)
pylab.show()


from sklearn.ensemble import RandomForestRegressor
X, y = [], []
# Load training data set

for params in train_dict:
    y.append(params)
    X.append(train_dict[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print(rfr.score(X, y), '\n')

"""
# for training on noise
for params in test_dict100:
    y.append(params)
    X.append(test_dict100[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print(rfr.score(X, y), '\n')

X_test, y_test = [], []
# Predict testing data set and print out results
for params in train_dict:
    y_test.append(params)
    X_test.append(train_dict[params].data)
    input = train_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
print(f"R2 no theta:  {rfr.score(X_test, y_test)}\n\n")
print()
"""

X_test, y_test = [], []
# Predict testing data set and print out results
for params in test_dict100:
    y_test.append(params)
    X_test.append(test_dict100[params].data)
    input = test_dict100[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
print(f"R2 theta = 100:  {rfr.score(X_test, y_test)}\n\n")
print()

X_test, y_test = [], []
# Predict testing data set and print out results
for params in test_dict1000:
    y_test.append(params)
    X_test.append(test_dict1000[params].data)
    input = test_dict1000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
print(f"R2 theta = 1000:  {rfr.score(X_test, y_test)}\n\n")
print()

X_test, y_test = [], []
for params in test_dict10000:
    y_test.append(params)
    X_test.append(test_dict10000[params].data)
    input = test_dict10000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
print(f"R2 theta = 10000:  {rfr.score(X_test, y_test)}\n\n")


