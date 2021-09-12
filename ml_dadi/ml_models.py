from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy import stats


def model_train(model, train_dict):
    # helper function for individual ML_train models
    # Load training data set from dictionary into arrays of input and
    # corresponding labels
    X_train_input = [train_dict[params].data.flatten()
                     for params in train_dict]
    y_train_label = [params for params in train_dict]
    # Output trained model
    return model.fit(X_train_input, y_train_label)


def mlpr_train(train_dict, mlpr=None, hidden_layer_sizes=(100,),
               activation='relu', solver='adam', alpha=0.1, max_iter=500):
    if mlpr is None:  # default MLPR setting if not specified
        mlpr = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation, solver=solver,
                            alpha=alpha, max_iter=max_iter)
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


def model_search(model, train_dict, param_grid, n_top=5):
    '''Use GridSearchCV to search for the best hyperparameters
    for ML models

    model: ML algorthims such as MLPR or RFR
    train_dict: dictionary of training data used for the tuning
    param_grid: dictionary of different model hyperparameters to be tuned
    n_top: int, number of top results to print out
    '''

    # load training data from train_dict
    train_features = [train_dict[params].data.flatten()
                      for params in train_dict]
    train_labels = [params for params in train_dict]

    # grid search
    cv = GridSearchCV(model, param_grid, n_jobs=-1)
    cv.fit(train_features, train_labels)
    results = cv.cv_results_

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def model_bootstrap(model, bootstrap_data):
    '''
    Note: replacing the bootstrap_predictions() method
    Runs the ML predictions for all of the bootstrapped FS
    trained_model is a single trained ML such as rfr_1000 used for prediction
    bootstrap_data is a list of dictionaries of 200 items with the format 
    {true_p:[orig_fs,[200 bootstrapped fs from orig_fs]]}
    for each theta case in the list.
    return: a list of bootstrap_pred dictionaries of the same
    format as bootstrap_data
    '''
    bootstrap_pred = []
    for theta_case in bootstrap_data:
        pred_dict = {}  # dictionary storing prediction results

        for true_p, all_fs in theta_case.items():
            # load elements from bootstrap data
            orig_fs = all_fs[0]  # single original fs
            # list of 200 bootstrapped fs from orig_fs
            list_bstr_fs = all_fs[1]

            # run prediction for the original fs
            orig_fs = orig_fs/orig_fs.sum()  # normalize for prediction
            orig_fs_pred = model.predict([orig_fs.flatten()]).flatten()

            # run prediction for fs bootstrapped from original fs
            list_bstr_fs_pred = []
            for bstr_fs in list_bstr_fs:
                bstr_fs = bstr_fs/bstr_fs.sum()  # normalize before prediction
                bstr_fs_pred = model.predict([bstr_fs.flatten()]).flatten()
                list_bstr_fs_pred.append(tuple(bstr_fs_pred))

            # save all predictions into a dictionary for each theta case
            pred_dict[true_p] = [tuple(orig_fs_pred), list_bstr_fs_pred]

        # save all dictionaries into a list for multiple theta cases
        bootstrap_pred.append(pred_dict)
    return bootstrap_pred

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
