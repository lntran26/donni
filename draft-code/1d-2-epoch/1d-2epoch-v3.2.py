### This program test the RFR algorithm with the second case
### where the training and testing data is with nu in log scale
### and tau range from 0-2, with and without normalization.

### Exactly the same as v3.py but without normalization step.

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
# change nu from 0.01 to 100, log scale 
for nu in [10**i for i in np.arange(-2, 2.1, 0.5)]:
    # change T from 0.5 to 2, increment by 0.5
    for T in np.arange(0.5, 2.1, 0.5):
        #params list for this spectrum
        params = (round(nu, 2), T)
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        # fsnorm = fs/fs.sum() # normalize all fs
        dict_train[params] = fs
# print(dict_train.keys())

# Generate testing data
# change nu=10^i, i=-1.9 to 2.1, step 0.8 
for nu in [10**i for i in np.arange(-1.7, 2, 0.6)]:
    # change T from 0.1 to 1.9, increment by 0.3
    for T in np.arange(0.1, 1.91, 0.3):
        #params list for this spectrum
        params = (round(nu, 2), round(T, 2))
        #generate spectrum
        fs = func_ex(params, ns, pts_l)
        # fsnorm = fs/fs.sum() # normalize all fs
        dict_test[params] = fs
# print(dict_test.keys())

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
for params in dict_test:
    input = dict_test[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))

# Check R2 score:
X_test, y_test = [], []
for params in dict_test:
    y_test.append(params)
    X_test.append(dict_test[params].data)
print('R2 score with test data: ', rfr.score(X_test, y_test))