from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
import numpy as np
from scipy import stats

def model_train(model, train_dict):
    # Load training data set from dictionary into arrays of input and
    # corresponding labels
    X_train_input = [train_dict[params].data.flatten() for params in train_dict]
    y_train_label = [params for params in train_dict]
    # Output trained model
    return model.fit(X_train_input, y_train_label)

def mlpr_train(train_dict, mlpr=None, solver='adam', max_iter=400):
    if mlpr is None: # default MLPR setting if not specified
        mlpr = MLPRegressor(solver=solver, max_iter=max_iter, alpha=0.01,
                        hidden_layer_sizes=(2000,), learning_rate='adaptive')
    return model_train(mlpr, train_dict)

def rfr_train(train_dict, ncpu=None):
    rfr = RandomForestRegressor(n_jobs=ncpu)
    return model_train(rfr, train_dict)

def vrfr_train(train_dict, ncpu=None):
    vrfr = ExtraTreesRegressor(n_jobs=ncpu)
    return model_train(vrfr, train_dict)

def model_test(model, test_dict, sort=False):
    """Test the performance of a trained ML model on test data set
    model: a trained scikit-learn ML model
    test_dict: test data set as a dictionary with true params as keys
    and sfs as corresponding values
    Output: Tuple of two lists, first list is true params, second list
    is params predicted by the trained model_name
    sort: if True will sort the true and pred values by params
    """
    y_true, y_pred = [], []
    for params in test_dict:
        # make a list of all true values
        # y_true is a list of param tuples
        y_true.append(params) 

        # make a list of all prediction values
        # first, we need to manipulate the form test data (SFS)
        test_fs = test_dict[params].data.flatten() 
        # test_dict[params] is a dadi spectrum object
        # .data change the spectrum object into a numpy array
        # .flatten() to convert into 1D array

        # next, we use the trained ML model to make prediction
        # and save into the list of predictions, y_pred
        y_pred.append(model.predict([test_fs]).flatten())
        # note: test_fs needs to be in a list form [test_fs]
        # for correct array dimension to be read by model.predict()
        # since scikit-learn predict() expects a 2D array object
    if sort: 
        # sort the lists by params instead of by test data set
        # before sorting:
            # y_true or y_pred = 
            #   [[p1_test1, p2_test1,...],...,[p1_test100, p2_test100,...],...]
        y_true = np.array(y_true).T.tolist()
        y_pred = np.array(y_pred).T.tolist()
        # after sorting:
            # y_true or y_pred = 
            # [[p1_test1,...,p1_test100,...],[p2_test1,p2_test100,...],..]      
    return y_true, y_pred

# Functions for scores
def r2(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    score_by_param = r2_score(y_true, y_pred, multioutput='raw_values')
    return score, score_by_param

def msle(y_true, y_pred):
    """msle only works for non-log values because values need to be non-neg"""
    score = mean_squared_log_error(y_true, y_pred)
    score_by_param = mean_squared_log_error(y_true, y_pred,
    multioutput='raw_values')
    return score, score_by_param

def rho(y_true, y_pred):
    """stats.spearmanr returns two values: correlation and p-value
    Here we only want the correlation value"""
    return stats.spearmanr(y_true, y_pred)[0]

# # Functions for MLPR
# def mlpr_train(train_dict, mlpr=None, std=False, solver='adam', max_iter=400):
#     # Load training data set from dictionary into arrays of input and
#     # corresponding labels
#     X_train_input = [train_dict[params].data.flatten() for params in train_dict]
#     y_train_label = [params for params in train_dict]

#     if mlpr is None: # default MLPR setting if not specified
#         mlpr = MLPRegressor(solver=solver, max_iter=max_iter, alpha=0.01,
#                         hidden_layer_sizes=(2000,), learning_rate='adaptive')

#     if std: # Perform data standardization step before training
#         scaler = preprocessing.StandardScaler().fit(X_train_input)
#         X_scaled = scaler.transform(X_train_input)
#         mlpr = mlpr.fit(X_scaled, y_train_label)
#     else: # Training without standardizing data
#         mlpr = mlpr.fit(X_train_input, y_train_label)
#     return mlpr

# def mlpr_test(mlpr, test_dict):
#     y_true, y_pred = [], []
#     for params in test_dict:
#         y_true.append(params)
#         test_fs = test_dict[params].data.flatten()
#         y_pred.append(mlpr.predict([test_fs]).flatten())
#     return y_true, y_pred

# # Functions for RFR
# def rfr_train(train_dict, ncpu=None):
#     # Load training data set from dictionary into arrays of input and
#     # corresponding labels
#     X_train_input = [train_dict[params].data.flatten() for params in train_dict]
#     y_train_label = [params for params in train_dict]

#     # Load RFR, specifying ncpu for parallel processing
#     rfr = RandomForestRegressor(n_jobs=ncpu)
#     # Train RFR
#     rfr = rfr.fit(X_train_input, y_train_label)
#     return rfr

# def rfr_test(rfr, test_dict):
#     y_true, y_pred = [], []
#     for params in test_dict:
#         y_true.append(params)
#         test_fs = test_dict[params].data.flatten()
#         y_pred.append(rfr.predict([test_fs]).flatten())
#     return y_true, y_pred
# def rfr_learn(train_dict, list_test_dict, ncpu=None):
#     '''
#     Trains a RandomForestRegressor algorithm and tests its performance.
#     Included argument ncpu for parallelism: default is None with ncpu=1; 
#     ncpu=-1 means using all available cpus. 
#     Returns a list of R2 scores measuring performance, 
#     which can be used to calculate average scores when running multiple
#     replicate experiments on the same training and testing conditions.
#     '''
#     # Load training data set from dictionary
#     X = [train_dict[params].data.flatten() for params in train_dict]
#     y = [params for params in train_dict]
    
#     # Load RFR, specifying ncpu for parallel processing
#     rfr = RandomForestRegressor(n_jobs=ncpu)
#     # rfr = RandomForestRegressor(criterion="poisson", n_jobs=ncpu)
#     # Train RFR
#     rfr = rfr.fit(X, y)
#     print('R2 score with train data:', rfr.score(X, y), '\n')

#     # Test RFR
#     score_list = []
#     count = 1 # Use count to print key# for each run
#     for test_dict in list_test_dict:
#         print('TEST CASE # ', str(count))
#         y_true, y_pred = [], []
#         for params in test_dict:
#             y_true.append(params)
#             test_fs = test_dict[params].data.flatten()
#             y_pred.append(rfr.predict([test_fs]).flatten())
#             print('Expected params: ', str(params), 
#                 ' vs. Predict params: ', str(rfr.predict([test_fs])))
#         score = mean_squared_log_error(y_true, y_pred)
#         score_list.append(score)
#         print('\n')
#         print('MSLE for each predicted param:', 
#                 mean_squared_log_error(y_true, y_pred, 
#                     multioutput='raw_values'))
#         print('Aggr. MSLE for all predicted params:', score)
#         print('R2 score for each predicted param:', 
#                     r2_score(y_true, y_pred, multioutput='raw_values'))
#         print('Aggr. R2 score for all predicted params:', 
#                     r2_score(y_true, y_pred),'\n')
#         count += 1
#     return score_list

# def rfr_learn_log(train_dict, list_test_dict, ncpu=None):
#     '''
#     Trains a RandomForestRegressor algorithm and tests its performance.
#     Included argument ncpu for parallelism: default is None with ncpu=1; 
#     ncpu=-1 means using all available cpus. 
#     Returns a list of R2 scores measuring performance, 
#     which can be used to calculate average scores when running multiple
#     replicate experiments on the same training and testing conditions.
#     This version include dealing with log transformed param
#     '''
#     # Training RFR
#     rfr = rfr_train(train_dict)

#     # Testing RFR
#     score_list = []
#     count = 1 # Use count to print key# for each run
#     for test_dict in list_test_dict:
#         print('TEST CASE # ', str(count))
#         y_true, y_pred = rfr_test(rfr, test_dict)
#         new_y_pred = un_log_transform_predict(y_pred, [0])
#         # if want the average MSLE
#         # score = mean_squared_log_error(y_true, new_y_pred)
#         # if want the average R^2
#         score = r2_score(y_true, new_y_pred)
#         score_list.append(score)
#         print('\n')
#         print('MSLE for each predicted param:', 
#                 mean_squared_log_error(y_true, new_y_pred, 
#                     multioutput='raw_values'))
#         print('Aggr. MSLE for all predicted params:', score)
#         print('R2 score for each predicted param:', 
#                     r2_score(y_true, new_y_pred, multioutput='raw_values'))
#         print('Aggr. R2 score for all predicted params:', 
#                     r2_score(y_true, new_y_pred),'\n')
#         count += 1
#     return score_list

# # Functions for Extra Trees
# def vrfr_train(train_dict, ncpu=None):
#     # Load training data set from dictionary into arrays of input and
#     # corresponding labels
#     X_train_input = [train_dict[params].data.flatten() for params in train_dict]
#     y_train_label = [params for params in train_dict]

#     # Load RFR, specifying ncpu for parallel processing
#     vrfr = ExtraTreesRegressor(n_jobs=ncpu)
#     # Train RFR
#     vrfr = vrfr.fit(X_train_input, y_train_label)
#     return vrfr

# def vrfr_test(vrfr, test_dict):
#     y_true, y_pred = [], []
#     for params in test_dict:
#         y_true.append(params)
#         test_fs = test_dict[params].data.flatten()
#         y_pred.append(vrfr.predict([test_fs]).flatten())
#     return y_true, y_pred

# def vrfr_learn(train_dict, list_test_dict, ncpu=None):
#     '''
#     Trains a RandomForestRegressor algorithm and tests its performance.
#     Included argument ncpu for parallelism: default is None with ncpu=1; 
#     ncpu=-1 means using all available cpus. 
#     Returns a list of R2 scores measuring performance, 
#     which can be used to calculate average scores when running multiple
#     replicate experiments on the same training and testing conditions.
#     '''
#     # Load training data set from dictionary
#     X = [train_dict[params].data.flatten() for params in train_dict]
#     y = [params for params in train_dict]
    
#     # Load RFR, specifying ncpu for parallel processing
#     vrfr = ExtraTreesRegressor(n_jobs=ncpu)
#     # Train RFR
#     vrfr = vrfr.fit(X, y)
#     print('R2 score with train data:', vrfr.score(X, y), '\n')

#     # Test RFR
#     score_list = []
#     count = 1 # Use count to print key# for each run
#     for test_dict in list_test_dict:
#         print('TEST CASE # ', str(count))
#         y_true, y_pred = [], []
#         for params in test_dict:
#             y_true.append(params)
#             test_fs = test_dict[params].data.flatten()
#             y_pred.append(vrfr.predict([test_fs]).flatten())
#             print('Expected params: ', str(params), 
#                 ' vs. Predict params: ', str(vrfr.predict([test_fs])))
#         score = mean_squared_log_error(y_true, y_pred)
#         score_list.append(score)
#         print('\n')
#         print('MSLE for each predicted param:', 
#                 mean_squared_log_error(y_true, y_pred, 
#                     multioutput='raw_values'))
#         print('Aggr. MSLE for all predicted params:', score)
#         print('R2 score for each predicted param:', 
#                     r2_score(y_true, y_pred, multioutput='raw_values'))
#         print('Aggr. R2 score for all predicted params:', 
#                     r2_score(y_true, y_pred),'\n')
#         count += 1
#     return score_list

# def vrfr_learn_log(train_dict, list_test_dict, ncpu=None):
#     '''
#     Trains a RandomForestRegressor algorithm and tests its performance.
#     Included argument ncpu for parallelism: default is None with ncpu=1; 
#     ncpu=-1 means using all available cpus. 
#     Returns a list of R2 scores measuring performance, 
#     which can be used to calculate average scores when running multiple
#     replicate experiments on the same training and testing conditions.
#     This version include dealing with log transformed param
#     '''
#     # Training RFR
#     vrfr = vrfr_train(train_dict)

#     # Testing RFR
#     score_list = []
#     count = 1 # Use count to print key# for each run
#     for test_dict in list_test_dict:
#         print('TEST CASE # ', str(count))
#         y_true, y_pred = vrfr_test(vrfr, test_dict)
#         new_y_pred = un_log_transform_predict(y_pred, [0])
#         # if want the average MSLE
#         # score = mean_squared_log_error(y_true, new_y_pred)
#         # if want the average R^2
#         score = r2_score(y_true, new_y_pred)
#         score_list.append(score)
#         print('\n')
#         print('MSLE for each predicted param:', 
#                 mean_squared_log_error(y_true, new_y_pred, 
#                     multioutput='raw_values'))
#         print('Aggr. MSLE for all predicted params:', score)
#         print('R2 score for each predicted param:', 
#                     r2_score(y_true, new_y_pred, multioutput='raw_values'))
#         print('Aggr. R2 score for all predicted params:', 
#                     r2_score(y_true, new_y_pred),'\n')
#         count += 1
#     return score_list