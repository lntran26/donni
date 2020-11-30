import os
import sys
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util
import pickle
from sklearn.ensemble import RandomForestRegressor

## This version train without first converting nu range into log scale!!

if __name__ == '__main__': 
    # load training set from pickle file
    list_train_dict = pickle.load(open('data/1d-2epoch/train-data-full','rb'))
    # list_train_dict = pickle.load(open('data/1d-2epoch/train-data-exclude','rb'))

    # train the random forest regressor with list of training sets
    # and save trained rfr as a list of rfr for each theta case
    list_rfr = [util.rfr_train(train_dict) for train_dict in list_train_dict]

    # save the list of trained rfr into pickle file
    pickle.dump(list_rfr, open('data/1d-2epoch/list-rfr-full', 'wb'), 2)
    # pickle.dump(list_rfr, open('data/1d-2epoch/list-rfr-exclude', 'wb'), 2)