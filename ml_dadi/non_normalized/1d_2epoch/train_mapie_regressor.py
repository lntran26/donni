import pickle
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
import os, sys
sys.path.append(os.path.join(os.getcwd(), '../..')) # this is the ml_dadi dir

# load data
train_data = pickle.load(open('data/train_data_10000','rb'))

# unpack train data set
# fs
X_train = [train_data[params].data.flatten()
           for params in train_data]
# params
y_train = [params for params in train_data]
# separate each param
N_train = [params[0] for params in y_train]
t_train = [params[1] for params in y_train]

# training mapie regressors
mlpr = MLPRegressor(hidden_layer_sizes=(500,),
                            activation='tanh', solver='lbfgs',
                            max_iter=10000)

# for N
mapie_N = MapieRegressor(mlpr)
mapie_N.fit(X_train, N_train)
# for t
mapie_t = MapieRegressor(mlpr)
mapie_t.fit(X_train, t_train)

pickle.dump(mapie_N, open('data/mapie_N', 'wb'), 2)
pickle.dump(mapie_t, open('data/mapie_t', 'wb'), 2)