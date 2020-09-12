### 09/11/2020

### This program train the RFR algorithm with the more dense
### input data by reducing the sparseness in the training set.
### Then we test with a more randomized training data.

### Train with noise, test with randomized noisy data

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
train_dict100 = {}
train_dict1000 = {}
train_dict10000 = {}

# change nu from 0.01 to 100, log scale 
for nu in [10**i for i in np.arange(-2, 2.1, 0.3)]:
    # change T from 0.1 to 2, increment by 0.1
    for T in np.arange(0.1, 2.1, 0.1):
        #params list for this spectrum
        params = (round(nu, 2), round(T, 1))
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        fsnorm = fs/fs.sum() # normalize all fs
        train_dict[params] = fsnorm

        #theta = 100
        fs100 = (100*fs).sample()
        fs100norm = fs100/fs100.sum()
        train_dict100[params] = fs100norm
        
        #theta = 1000
        fs1000 = (1000*fs).sample()
        fs1000norm = fs1000/fs1000.sum()
        train_dict1000[params] = fs1000norm

        #theta = 10000
        fs10000 = (10000*fs).sample()
        fs10000norm = fs10000/fs10000.sum()
        train_dict10000[params] = fs10000norm
        
# construct testing dictionary
test_dict = {}
test_dict100 = {}
test_dict1000 = {}
test_dict10000 = {}

# generate 100 random tests within range
for i in range(50):
    # generate random nu and T
    nu = 10 ** (random.random() * 4 - 2)
    T = random.random() * 1.9 + 0.1
    params = (round(nu, ndigits=2), round(T, ndigits=1))
    fs = func_ex(params, ns, pts_l)
    fsnorm = fs/fs.sum() # normalize all fs
    test_dict[params] = fsnorm

    #theta = 100
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

import os
import sys

sys.stdout = open("random-noise-noise-out.txt", "a")

print('Experiment date: ', time.asctime(time.localtime(time.time())))
### FIRST SUBCASE: Train on theta=100
print('FIRST SUBCASE: Train on theta=100')
from sklearn.ensemble import RandomForestRegressor
X, y = [], []
# Load training data set for theta = 100
for params in train_dict100:
    y.append(params)
    X.append(train_dict100[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print('R2 score with train data/fit: ', rfr.score(X, y), '\n')

# Predict testing data set and print out results
## TEST CASES:
## Test on test data without noise
for params in test_dict:
    input = test_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict:
    y_test.append(params)
    X_test.append(test_dict[params].data)
print(f"R2 test no noise:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=100
for params in test_dict100:
    input = test_dict100[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict100:
    y_test.append(params)
    X_test.append(test_dict100[params].data)
print(f"R2 test theta=100:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=1000
for params in test_dict1000:
    input = test_dict1000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict1000:
    y_test.append(params)
    X_test.append(test_dict1000[params].data)
print(f"R2 test theta=1000:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=10000
for params in test_dict10000:
    input = test_dict10000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict10000:
    y_test.append(params)
    X_test.append(test_dict10000[params].data)
print(f"R2 test theta=10000:  {rfr.score(X_test, y_test)}\n\n")

## Test on training data without noise
for params in train_dict:
    input = train_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in train_dict:
    y_test.append(params)
    X_test.append(train_dict[params].data)
print(f"R2 no theta same train params:  {rfr.score(X_test, y_test)}\n\n")
print()

### SECOND SUBCASE: Train on theta=1000
print('SECOND SUBCASE: Train on theta=1000')
from sklearn.ensemble import RandomForestRegressor
X, y = [], []
# Load training data set for theta = 1000
for params in train_dict1000:
    y.append(params)
    X.append(train_dict1000[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print('R2 score with train data/fit: ', rfr.score(X, y), '\n')

# Predict testing data set and print out results
## TEST CASES:
## Test on test data without noise
for params in test_dict:
    input = test_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict:
    y_test.append(params)
    X_test.append(test_dict[params].data)
print(f"R2 test no noise:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=100
for params in test_dict100:
    input = test_dict100[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict100:
    y_test.append(params)
    X_test.append(test_dict100[params].data)
print(f"R2 test theta=100:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=1000
for params in test_dict1000:
    input = test_dict1000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict1000:
    y_test.append(params)
    X_test.append(test_dict1000[params].data)
print(f"R2 test theta=1000:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=10000
for params in test_dict10000:
    input = test_dict10000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict10000:
    y_test.append(params)
    X_test.append(test_dict10000[params].data)
print(f"R2 test theta=10000:  {rfr.score(X_test, y_test)}\n\n")

## Test on training data without noise
for params in train_dict:
    input = train_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in train_dict:
    y_test.append(params)
    X_test.append(train_dict[params].data)
print(f"R2 no theta same train params:  {rfr.score(X_test, y_test)}\n\n")
print()

### THIRD SUBCASE: Train on theta=10000
print('THIRD SUBCASE: Train on theta=10000')
from sklearn.ensemble import RandomForestRegressor
X, y = [], []
# Load training data set for theta = 10000
for params in train_dict10000:
    y.append(params)
    X.append(train_dict10000[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print('R2 score with train data/fit: ', rfr.score(X, y), '\n')

# Predict testing data set and print out results
## TEST CASES:
## Test on test data without noise
for params in test_dict:
    input = test_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict:
    y_test.append(params)
    X_test.append(test_dict[params].data)
print(f"R2 no noise:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=100
for params in test_dict100:
    input = test_dict100[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict100:
    y_test.append(params)
    X_test.append(test_dict100[params].data)
print(f"R2 test theta=100:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=1000
for params in test_dict1000:
    input = test_dict1000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict1000:
    y_test.append(params)
    X_test.append(test_dict1000[params].data)
print(f"R2 test theta=1000:  {rfr.score(X_test, y_test)}\n\n")

## Test on noisy data theta=10000
for params in test_dict10000:
    input = test_dict10000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict10000:
    y_test.append(params)
    X_test.append(test_dict10000[params].data)
print(f"R2 test theta=10000:  {rfr.score(X_test, y_test)}\n\n")

## Test on training data without noise
for params in train_dict:
    input = train_dict[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in train_dict:
    y_test.append(params)
    X_test.append(train_dict[params].data)
print(f"R2 no theta same train params:  {rfr.score(X_test, y_test)}\n\n")
print()

sys.stdout.close()
sys.exit()