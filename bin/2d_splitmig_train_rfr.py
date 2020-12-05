import os
import sys
import dadi
# specify the path to util.py file
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util
import pickle
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__': 
    # load training set from pickle file
    list_train_dict = pickle.load(open('data/2d-splitmig/train-data','rb'))
    
    # sample to make 10,000 n_samples total for each theta cases, 
    # train the random forest regressor with list of training sets,
    # and save trained rfr as a list of rfr for each theta case
    list_rfr = []
    for train_dict in list_train_dict:
        X_train_input, y_train_label = [], []
        params_list = list(train_dict.keys())
                
        for params in params_list:
            # load the original params:fs pair and append into data lists
            fs = train_dict[params]
            y_train_label.append(params)
            X_train_input.append(fs)
            for i in range(3):
                # sample 3x for each params:fs pair and append to data lists
                fs_tostore = abs(fs).sample()
                y_train_label.append(params)
                X_train_input.append(fs_tostore)

        # need to convert fs from dadi spectrum object to numpy array
        # before can be read into the RFR for training
        X_train_input = [fs.data.flatten() for fs in X_train_input]

        # Train RFR with the X and y data sets after sampling
        rfr = RandomForestRegressor()
        rfr = rfr.fit(X_train_input, y_train_label)
        
        # Save each trained rfr for each theta case
        list_rfr.append(rfr)

    # # this version doesn't include sampling
    # list_rfr = [util.rfr_train(train_dict) for train_dict in list_train_dict]

    # save the list of trained rfr into pickle file
    pickle.dump(list_rfr, open('data/2d-splitmig/list-rfr-sampling', 'wb'), 2)
    # pickle.dump(list_rfr, open('data/2d-splitmig/list-rfr', 'wb'), 2)