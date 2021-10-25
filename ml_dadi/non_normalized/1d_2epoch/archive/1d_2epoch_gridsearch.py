import pickle
import time
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))  # this is the ml_dadi dir
import ml_models
from ml_models import model_search
from sklearn.neural_network import MLPRegressor


# Load training data
train_dict = pickle.load(open('data/1d_2epoch/train_data_10000', 'rb'))

# Specify the ML models to be optimized
mlpr = MLPRegressor()

# Specify param_grid to do the grid search over
param_grid = {'hidden_layer_sizes': [(100,), (500,), (1000,)],
              'activation':  ['relu'],
              'solver': ['lbfgs', 'adam'],
              'max_iter': [5000]
              }
# open a text file to record experiment results
timestr = time.strftime("%Y%m%d-%H%M%S")
sys.stdout = open(
    'results/gridsearch/1d_2epoch_gridsearch_' + timestr + '.txt', 'w')
for key, value in param_grid.items():
    print(key, ':', value, "\n")

model_search(mlpr, train_dict, param_grid, n_top=36)
# close the text file
sys.stdout.close()
# exit mode
sys.exit()
