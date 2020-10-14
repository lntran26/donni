# Collection of functions that can be used to generate ML data sets
# and train/test RFR algorithm w params of different demographic models.
# Functions include parallelized options for multiprocessing.

from multiprocessing import Pool
import dadi
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Technical details for working with multiprocessing Pools:
# First, they only work with single-argument functions.
# Second, you can't pass newly-created functions to them. So we need to do
# the make_extrap_func inside it here.
def worker_func(args):
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)

def generating_data_parallel(params_list, theta_list, 
                                func, ns, pts_l, ncpu=None):
    '''Parallelized version for generating_data using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    Returns a list of dictionaries where each dictionary stores
    a data set for training or testing RFR algorithm. Dictionaries
    have structure params:fs.
    '''
    arg_list = [(p, func, ns, pts_l) for p in params_list]
    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(worker_func, arg_list)

    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            if theta == 1:
                fs_tostore = fs
            else:
                fs_tostore = (theta*abs(fs)).sample()
            if fs_tostore.sum()==0:
                pass
            else:    
                data_dict[params] = fs_tostore/fs_tostore.sum()                
        list_dicts.append(data_dict)
    return list_dicts

def rfr_learn(train_dict, list_test_dict, ncpu=None):
    '''
    Trains a RandomForestRegressor algorithm and tests its performance.
    Included argument ncpu for parallelism: default is None with ncpu=1; 
    ncpu=-1 means using all available cpus. 
    Returns a list of R2 scores measuring performance, 
    which can be used to calculate average scores when running multiple
    replicate experiments on the same training and testing conditions.
    '''
    # Load training data set from dictionary
    X = [train_dict[params].data.flatten() for params in train_dict]
    y = [params for params in train_dict]
    
    # Load RFR, specifying ncpu for parallel processing
    rfr = RandomForestRegressor(n_jobs=ncpu)
    # Train RFR
    rfr = rfr.fit(X, y)
    print('R2 score with train data:', rfr.score(X, y), '\n')

    # Test RFR
    score_list = []
    count = 1 # Use count to print key# for each run
    for test_dict in list_test_dict:
        print('TEST CASE # ', str(count))
        y_true, y_pred = [], []
        for params in test_dict:
            y_true.append(params)
            test_fs = test_dict[params].data.flatten()
            y_pred.append(rfr.predict([test_fs]).flatten())
            print('Expected params: ', str(params), 
                ' vs. Predict params: ', str(rfr.predict([test_fs])))
        score = mean_squared_log_error(y_true, y_pred)
        score_list.append(score)
        print('\n')
        print('MSLE for each predicted param:', 
                mean_squared_log_error(y_true, y_pred, 
                    multioutput='raw_values'))
        print('Aggr. MSLE for all predicted params:', score)
        print('R2 score for each predicted param:', 
                    r2_score(y_true, y_pred, multioutput='raw_values'))
        print('Aggr. R2 score for all predicted params:', 
                    r2_score(y_true, y_pred),'\n')
        count += 1
    return score_list

def rfr_train(train_dict, ncpu=None):
    # Load training data set from dictionary into arrays of input and
    # corresponding labels
    X_train_input = [train_dict[params].data.flatten() for params in train_dict]
    y_train_label = [params for params in train_dict]

    # Load RFR, specifying ncpu for parallel processing
    rfr = RandomForestRegressor(n_jobs=ncpu)
    # Train RFR
    rfr = rfr.fit(X_train_input, y_train_label)
    return rfr

def rfr_test(rfr, test_dict):
    y_true, y_pred = [], []
    for params in test_dict:
        y_true.append(params)
        test_fs = test_dict[params].data.flatten()
        y_pred.append(rfr.predict([test_fs]).flatten())
    return y_true, y_pred

def rfr_r2_score(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    score_by_param = r2_score(y_true, y_pred, multioutput='raw_values')
    return score, score_by_param

def rfr_msle(y_true, y_pred):
    score = mean_squared_log_error(y_true, y_pred)
    score_by_param = mean_squared_log_error(y_true, y_pred,
    multioutput='raw_values')
    return score, score_by_param

def sort_by_param(y_true, y_pred):
    '''
    Sort the output of rfr_test into lists of true vs predict values by each 
    param used in the model
    Returns: param_true and param_pred are each a list of lists, each sublist
    contains values for one param
    '''
    param_true, param_pred = [], []
    n=0
    while n < len(y_true[n]):
        param_list_true, param_list_pred = [], []
        for true, pred in zip(y_true, y_pred):
            param_list_true.append(true[n])
            param_list_pred.append(pred[n])
        param_true.append(param_list_true)
        param_pred.append(param_list_pred)
        n+=1
    return param_true, param_pred

def plot_by_param(true, pred, r2, msle, ax=None):
    '''
    true, pred = list of true and predicted values for one param,
    which can be obtained from sort_by_param;
    r2: one r2 score for one param of one train:test pair
    msle: one msle score for one param of one train:test pair
    '''
    if ax is None:
        ax = plt.gca()
    # only plot in log scale if the difference between max and min is large
    if max(true+pred)/min(true+pred) > 100:
        plt.xscale("log")
        plt.yscale("log")
    else:
        pass
    plot = plt.scatter(true, pred)
    plt.xlabel("true")
    plt.ylabel("predicted")
    (m,b) = np.polyfit(true, pred, 1)
    plt.plot(np.unique(true), np.poly1d((m,b))(np.unique(true)), color='red')
    equation = 'y = ' + str(round(m,4)) + 'x' ' + ' + str(round(b,4))
    plt.text(0.2, 0.9, equation + "\nR^2: " + str(round(r2,4))+ "\nMSLE: "
    + str(round(msle,4)), horizontalalignment='center',
    verticalalignment='center', transform = ax.transAxes)
    return plot

# Test code: running time for the 1D version (2 epoch)
# We protect this test code with this Python idiom. This means the test
# code won't run when we "import util", which is useful for defining
# functions we'll want to use in multiple scripts.
# if __name__ == "__main__":
#     import time
#     import numpy as np
#     import random
#     # Generate test arguments. Note fancy list comprehension usage here.
#     # Also, np.linspace is often easier to use than arange.
#     train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 10)
#                           for T in np.linspace(0.1,2,10)]

#     test_params = []
#     for i in range(100):
#         nu = 10 ** (random.random() * 4 - 2)
#         T = random.random() * 1.9 + 0.1
#         params = (round(nu, 2), round(T, 1))
#         test_params.append(params)
    
#     theta_list = [1, 1000]
#     func = dadi.Demographics1D.two_epoch
#     ns = [20]
#     pts_l = [40, 50, 60]

#     # testing running time for the generating_data function
#     start = time.time()
#     generating_data(train_params, theta_list, func, ns, pts_l)
#     print('Serial execution time to generate data 1D: {0:.2f}s'
#     .format(time.time()-start))

#     start = time.time()
#     generating_data_parallel(train_params, theta_list, func, ns, pts_l)
#     print('Parallel execution time to generate data 1D: {0:.2f}s'
#     .format(time.time()-start))

    # Generating data for RFR learning
    # list_train_dict = generating_data_parallel(train_params, 
    #                         theta_list, func, ns, pts_l)
    # list_test_dict = generating_data_parallel(test_params, 
    #                         theta_list, func, ns, pts_l)

    # # testing running time for the rfr_learn function
    # start = time.time()
    # for train_dict in list_train_dict:
    #     rfr_learn(train_dict, list_test_dict)
    # print('Serial execution time to learn 1D: {0:.2f}s'
    # .format(time.time()-start))

    # start = time.time()
    # for train_dict in list_train_dict:
    #     rfr_learn(train_dict, list_test_dict, -1)
    # print('Parallel execution time to learn 1D: {0:.2f}s'
    # .format(time.time()-start))

# Test code: running time for the 2D version (split_mig)
# if __name__ == "__main__":
#     import time
#     import numpy as np
#     import random

#     # generate training params list
#     train_params = [(nu1, nu2, T, m) for nu1 in 10**np.linspace(-2, 2, 3)
#                                 for nu2 in 10**np.linspace(-2, 2, 3)
#                                 for T in np.linspace(0.1, 2, 3)
#                                 for m in np.linspace(1, 10, 3)]

#     # generate testing params list
#     test_params = []
#     for i in range(50):
#     # generate random nu and T within the same range as training data range
#         nu1 = 10 ** (random.random() * 4 - 2)
#         nu2 = 10 ** (random.random() * 4 - 2)
#         T = random.random() * 1.9 + 0.1
#         m = random.random() * 9.9 + 0.1
#         params = (round(nu1, 2), round(nu2,2), round(T, 1), round(m, 1))
#         test_params.append(params)
    
#     theta_list = [1,100] # theta_list[1,1000] currently gives error
#     func = dadi.Demographics2D.split_mig
#     ns = [20,20]
#     pts_l = [40, 50, 60]

    # testing running time for the generating_data function
    # start = time.time()
    # generating_data(train_params, theta_list, func, ns, pts_l)
    # print('Serial execution time to generate data 2D: {0:.2f}s'
    # .format(time.time()-start))

    # start = time.time()
    # generating_data_parallel(train_params, theta_list, func, ns, pts_l)
    # print('Parallel execution time to generate data 2D: {0:.2f}s'
    # .format(time.time()-start))

    # Generating data for RFR learning
    # list_train_dict = generating_data_parallel(train_params, theta_list, 
    #                             func, ns, pts_l)
    # list_test_dict = generating_data_parallel(test_params, theta_list, 
    #                             func, ns, pts_l)

    # testing running time for the rfr_learn function
    # start = time.time()
    # for train_dict in list_train_dict:
    #     rfr_learn(train_dict, list_test_dict)
    # print('Serial execution time to learn 2D: {0:.2f}s'
    # .format(time.time()-start))

    # start = time.time()
    # for train_dict in list_train_dict:
    #     rfr_learn(train_dict, list_test_dict, -1)
    # print('Parallel execution time to learn 2D: {0:.2f}s'
    # .format(time.time()-start))

# Test code: plotting 1D example with nu
if __name__ == "__main__":
    import random
    # generate training and testing params values
    train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 21)
                          for T in np.linspace(0.1,2,20)]
    test_params = []
    for i in range(50):
        nu = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        params = (round(nu, 2), round(T, 1))
        test_params.append(params)
    # designate theta, demographic model, sample size, and extrapolation grid
    theta_list = [1]
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]
    # make training and testing datasets
    list_train_dict = generating_data_parallel(train_params,
                        theta_list, func, ns, pts_l)
    list_test_dict = generating_data_parallel(test_params,
                        theta_list, func, ns, pts_l)
    # train rfr
    rfr = rfr_train(list_train_dict[0], -1)
    # test rfr
    y_true, y_predict = rfr_test(rfr, list_test_dict[0])
    # plot results for 1 parameter nu
    nu_true = sort_by_param(y_true, y_predict)[0][0]
    nu_pred = sort_by_param(y_true, y_predict)[1][0]
    nu_r2 = rfr_r2_score(y_true, y_predict)[1][0]
    nu_msle = rfr_msle(y_true, y_predict)[1][0]
    plot_by_param(nu_true, nu_pred, nu_r2, nu_msle)
    a = str(1)
    # output figure will be saved to current working directory
    plt.savefig('fig'+a+'.pdf')
    plt.clf()

# Test code: plotting 2D example with m
if __name__ == "__main__":
    import random
    # generate training and testing params values
    train_params = [(nu1, nu2, T, m) for nu1 in 10**np.linspace(-2, 2, 3)
                                    for nu2 in 10**np.linspace(-2, 2, 3)
                                    for T in np.linspace(0.1, 2, 3)
                                    for m in np.linspace(1, 10, 3)]
    test_params = []
    for i in range(50):
    # generate random nu and T within the same range as training data range
        nu1 = 10 ** (random.random() * 4 - 2)
        nu2 = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        m = random.random() * 9.9 + 0.1
        params = (round(nu1, 2), round(nu2, 2), round(T, 1), round(m, 1))
        test_params.append(params)
    # designate theta, demographic model, sample size, and extrapolation grid
    theta_list = [1]
    func = dadi.Demographics2D.split_mig
    ns = [20,20]
    pts_l = [40, 50, 60]
    # make training and testing datasets
    list_train_dict = generating_data_parallel(train_params,
                        theta_list, func, ns, pts_l)
    list_test_dict = generating_data_parallel(test_params,
                        theta_list, func, ns, pts_l)
    # train rfr
    rfr = rfr_train(list_train_dict[0], -1)
    # test rfr
    y_true, y_predict = rfr_test(rfr, list_test_dict[0])
    # plot results for 1 parameter m
    m_true = sort_by_param(y_true, y_predict)[0][3]
    m_pred = sort_by_param(y_true, y_predict)[1][3]
    m_r2 = rfr_r2_score(y_true, y_predict)[1][3]
    m_msle = rfr_msle(y_true, y_predict)[1][3]
    plot_by_param(m_true, m_pred, m_r2, m_msle)
    a = str(2)
    # output figure will be saved to current working directory
    plt.savefig('fig'+a+'.pdf')
    plt.clf()