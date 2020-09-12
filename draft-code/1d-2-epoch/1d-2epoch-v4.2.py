### This program test the RFR algorithm with the third case
### where the training and testing data is with nu in log scale
### and tau range from 0-2. We train the algorithm by non-noise
### data then test with noisy data.

### Exactly the same as v4 but test with a subcase where
### the test parameter values are different from the training
### parameter values.

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
test_dict = {}
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
        
# print(train_dict.keys())

for nu in [10**i for i in np.arange(-1.7, 2, 0.6)]:
    # change T from 0.1 to 1.9, increment by 0.3
    for T in np.arange(0.1, 1.91, 0.3):
        #params list for this spectrum
        params = (round(nu, 2), round(T, 2))
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        fsnorm = fs/fs.sum() # normalize all fs
        test_dict[params] = fsnorm        
        
        #create noisy data for testing with different params 
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

# print(test_dict.keys())
# print(test_dict10000.keys())

from sklearn.ensemble import RandomForestRegressor
X, y = [], []
# Load training data set
for params in train_dict:
    y.append(params)
    X.append(train_dict[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print('R2 score with train data/fit: ', rfr.score(X, y))

# Predict testing data set and print out results
for params in test_dict100:
    input = test_dict100[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict100:
    y_test.append(params)
    X_test.append(test_dict100[params].data)
print(f"R2 theta = 100:  {rfr.score(X_test, y_test)}\n\n")

for params in test_dict1000:
    input = test_dict1000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict1000:
    y_test.append(params)
    X_test.append(test_dict1000[params].data)
print(f"R2 theta = 1000:  {rfr.score(X_test, y_test)}\n\n")

for params in test_dict10000:
    input = test_dict10000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict10000:
    y_test.append(params)
    X_test.append(test_dict10000[params].data)
print(f"R2 theta = 10000:  {rfr.score(X_test, y_test)}\n\n")

