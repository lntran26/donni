### This program test the RFR algorithm with the fourth case
### where we train with noisy data and test with noisy data
### Two different sets of parameter values will be used.

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
train_dict100 = {}
train_dict1000 = {}
train_dict10000 = {}
test_dict = {}
test_dict100 = {}
test_dict1000 = {}
test_dict10000 = {}

# Generating training data with noise
# change nu from 0.01 to 100, log scale
for nu in [10**i for i in np.arange(-2, 2.1, 0.5)]:
    # change T from 0.5 to 2, increment by 0.5
    for T in np.arange(0.5, 2.1, 0.5):
        #params list for this spectrum
        params = (round(nu, 2), round(T, 1))
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        # fs = fs.sample()
        fsnorm = fs/fs.sum()
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

# Generating testing data with noise
for nu in [10**i for i in np.arange(-1.7, 2, 0.6)]:
    # change T from 0.1 to 1.9, increment by 0.3
    for T in np.arange(0.1, 1.91, 0.3):
        #params list for this spectrum
        params = (round(nu, 2), round(T, 2))
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        fsnorm = fs/fs.sum()
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

# print(train_dict100.keys())
# print(test_dict10000.keys())
# print(test_dict.keys())

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

##------------------------------------------------------
# ### plotting the models 
# fs = train_dict[(1.0, 0.5)]
# dadi.Plotting.plot_1d_fs(fs)
# pylab.show()

# fs = train_dict100[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs)
# pylab.show()

# fs = train_dict1000[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs)
# pylab.show()

# fs = train_dict10000[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs)
# pylab.show()



# #print(test_dict100[(1, 0.5)], '\n')
# fs100 = train_dict100[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs100)
# pylab.show()


# #print(test_dict10000[(1, 0.5)])
# fs10000 = train_dict10000[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs10000)
# pylab.show()

# fs = test_dict[(5.01, 0.4)]
# dadi.Plotting.plot_1d_fs(fs)
# pylab.show()

# fs = test_dict10000[(5.01, 0.4)]
# dadi.Plotting.plot_1d_fs(fs)
# pylab.show()

# #print(test_dict100[(1, 0.5)], '\n')
# fs100 = test_dict100[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs100)
# pylab.show()

# #print(test_dict10000[(1, 0.5)])
# fs10000 = test_dict10000[(1, 0.5)]
# dadi.Plotting.plot_1d_fs(fs10000)
# pylab.show()