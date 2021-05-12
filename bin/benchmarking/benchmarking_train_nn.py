import os
import sys
import dadi
import util
import pickle
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__': 
    # load training set from pickle file
    list_train_dict = pickle.load(open('data/2d-splitmig/train-data','rb'))
    
    # train the MLP regressor with list of training sets,
    # and save trained nn as a list of nn for each theta case
    list_nn = []
    for train_dict in list_train_dict:
        X_train_input, y_train_label = [], []
        params_list = list(train_dict.keys())
                
        for params in params_list:
            # load the original params:fs pair and append into data lists
            fs = train_dict[params]
            y_train_label.append(params)
            X_train_input.append(fs)

        # need to convert fs from dadi spectrum object to numpy array
        # before can be read into the RFR for training
        X_train_input = [fs.data.flatten() for fs in X_train_input]

        # Train NN for each theta case
        nn = MLPRegressor(solver = 'adam', max_iter=400, alpha=0.001,
                        hidden_layer_sizes=(2000,), learning_rate='adaptive')
        nn = nn.fit(X_train_input, y_train_label)
        
        # Save each trained nn for each theta case
        list_nn.append(nn)

    # save the list of trained nn into pickle file
    pickle.dump(list_nn, open('data/2d-splitmig/list-nn', 'wb'), 2)