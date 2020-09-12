### This program aims to produce SFS with varying parameters
### for the 2-epoch, 1D model and save all resulting SFS into
### a dictionary file along with the generating parameter values
### so it could be loaded into a Random Forest ML algorithm for
### training. We then test the trained function using a different
### testing dataset.
import numpy as np
import dadi

# Create the extrapolated version of the model function
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_func(func)

# Grid point settings for extrapolation.
pts_l = [40,50,60]
# sample size
ns = [20,]

# Initialize empty dictionary to populate with keys:params and values:fs
dict_train = {}
dict_test = {}

# Generate training data of params and fs
for nu in np.arange(0.1, 5, round(0.5, ndigits=1)):
    for tau in np.arange(0, 1.1, round(0.1, ndigits=1)):
        # generate a list of params to make fs
        params = (round(nu, ndigits=1), round(tau, ndigits=1))
        # generate a spectrum for each set of params
        fs = func_ex(params, ns, pts_l)
        # save params and fs pairs into a dictionary
        dict_train[params] = fs

# print(dict_train.keys())
# Generate testing data
for nu in np.arange(0.2, 5.1, round(0.3, ndigits=1)):
    for tau in np.arange(0, 1.1, round(0.2, ndigits=1)):
        # generate a list of params to make fs
        params = (round(nu, ndigits=1), round(tau, ndigits=1))
        # generate a spectrum for each set of params
        fs = func_ex(params, ns, pts_l)
        # save params and fs pairs into a dictionary
        dict_test[params] = fs

# print(dict_test.keys())
# dict_test[(0.2, 0.0)].data

### Train RandomForestRegressor algorithm
# Import scikit-learn library
from sklearn.ensemble import RandomForestRegressor

# Initialize training data array
X , y = [], [] 
# Load training data set
for params in dict_train:
    y.append(params)
    X.append(dict_train[params].data)
# Fit regression model
rfr = RandomForestRegressor()
rfr = rfr.fit(X, y)
print('R2 score with train data/fit: ', rfr.score(X, y))

# Predict testing data set and print out results
# import os
# import sys
# sys.stdout = open("test.txt", "w")
for params in dict_test:
    input = dict_test[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
# sys.stdout.close()

# sys.stdout = open("test.txt", "a")
# print(dict_train.keys())
# sys.stdout.close()

# have to exit the program to do other things
# sys.exit()

# Connie adds:
# stdout_obj = sys.stdout to open
# then to close I had to do
# sys.stdout = stdout_obj
# since she's using IDE, the program wouldn't stop running
# if not do the above
# Also, if want to save multiple runs to the same text file, 
# you can use "a" (append) instead of "w" (write over)

# # The ML didn't do very well at tau = 0 --> examine the graph
# # --> It's probably the same for all graph at tau = 0
# params = [4, 0]
# fs = func_ex(params, ns, pts_l)
# dadi.Plotting.plot_1d_fs(fs)

# Check R2 score:
X_test, y_test = [], []
for params in dict_test:
    y_test.append(params)
    X_test.append(dict_test[params].data)
print('R2 score with test data: ', rfr.score(X_test, y_test))
