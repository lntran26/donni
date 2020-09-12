"""
Created on Fri Sep  4 09:52:06 2020

@author: conniesun
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
test_dict100 = {}
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
        
        #create noisy data for testing 
        #theta = 100
        fs100 = (100*fs).sample()
        fs100norm = fs100/fs100.sum()
        test_dict100[params] = fs100norm
        
        #theta = 10000
        fs10000 = (10000*fs).sample()
        fs10000norm = fs10000/fs10000.sum()
        test_dict10000[params] = fs10000norm

# #plotting the models 
# fs = train_dict[(1, 0.5)]
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
print()


for params in test_dict10000:
    input = test_dict10000[params].data
    print('Expected params: ', str(params), 
        ' vs. Predict params: ', str(rfr.predict([input])))
X_test, y_test = [], []
for params in test_dict10000:
    y_test.append(params)
    X_test.append(test_dict10000[params].data)
print(f"R2 theta = 10000:  {rfr.score(X_test, y_test)}\n\n")

