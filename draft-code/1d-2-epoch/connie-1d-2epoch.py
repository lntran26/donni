import dadi
import pickle

# 1 population, two epoch model
func = dadi.Demographics1D.two_epoch
func_ex = dadi.Numerics.make_extrap_func(func)

ns = [20,]
pts_l = [40, 50, 60]

dict = {}

nu = 0.5
# change nu from 0.5 to 4.5, increment by 1
while nu <= 4.5:
    T = 0.5
    # change T from 0.5 to 2.5, increment by 1
    while T <= 2.5:
        # params list for this spectrum
        params = (nu, T)
        # generate spectrum
        fs = func_ex(params, ns, pts_l)
        dict[params] = fs
        T += 1
    nu += 1

# use pickle.dump to save the dictionary as a pickle file
pickle.dump(dict, open('pickle_fs_two_epoch', 'wb'), 2)

# create and save a "testing" dict for ML algorithm
test_params = [(2.5, 1.5),(3.0, 0.8),(1.2, 0.3)]
test_dict = {}
for params in test_params:
    fs = func_ex(params, ns, pts_l)
    test_dict[params] = fs
pickle.dump(test_dict, open('pickle_fs_two_epoch_test', 'wb'), 2)

## ML
from sklearn.ensemble import RandomForestRegressor
import pickle

dict = pickle.load(open('pickle_fs_two_epoch','rb'))

X , y = [], []

for params in dict:
    y.append(params)
    X.append(dict[params].data)
regr = RandomForestRegressor()
regr.fit(X, y)

# testing
test_dict = pickle.load(open('pickle_fs_two_epoch_test', 'rb'))
# params nu = 2.5, T = 1.5
# use the spectrum produced by dadi that was actually used
# in the training data and see if output matches
input = test_dict[(2.5, 1.5)].data
# predict
print("Expected params: [2.5, 1.5] -- ML: " + str(regr.predict([input])))

# params nu = 3.0, T = 0.8
# this is the spectrum produced by dadi with these parameters
# give as input and see if output matches
input = test_dict[(3.0, 0.8)].data
# predict
print("Expected params: [3.0, 0.8] -- ML: " + str(regr.predict([input])))

# params nu = 1.2, T = 0.3
# give time input that is outside of training data bounds
input = test_dict[(1.2, 0.3)].data
print("Expected params: [1.2, 0.3] -- ML: " + str(regr.predict([input])))