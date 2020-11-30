import dadi
import numpy as np
import random
# specify the path to util.py file
import os
import sys
sys.path.insert(1, os.path.join(os.getcwd(), 'bin'))
import util
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # generate parameter list for testing
    test_params = []
    # range(#) dictate how many values are in each test set
    for i in range(50):
    # generate random nu and T within the same range as training data range
        nu = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        params = (nu, T)
        # params = (round(nu, 2), round(T, 1))
        test_params.append(params)
    # print testing set info 
    print('n_samples testing: ', len(test_params))
    print('Range of testing params:', min(test_params), 'to', 
            max(test_params))
  
    # designate demographic model, sample size, and extrapolation grid 
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # # Generating data the regular way
    # # construct testing dictionary with structure params:fs
    # test_dict = {}
    # list_test_dict=[]
    # func_ex = dadi.Numerics.make_extrap_func(func)
    # for params in test_params:
    #     test_dict[params] = func_ex(params, ns, pts_l)
    # list_test_dict.append(test_dict)
    # theta_list = [1]

    # generate a list of theta values to run scaling and add variance
    # theta_list = [1, 100, 1000, 10000]
    theta_list = [1,1000]
    print('Theta list:', theta_list)

    # Use function to make lists of dictionaries storing different
    # testing data sets from lists of parameters
    list_test_dict = util.generating_data_parallel(test_params, 
                        theta_list, func, ns, pts_l)

    # This is our initial guess for the parameters, which is somewhat arbitrary.
    p0 = [2, 0.2]
    upper_bound = [100, 2]
    lower_bound = [1e-2, 0.1]
    grids = pts_l
    i=0
    count_test = 1
    for test_dict in list_test_dict:
        y_true, y_pred = [], []
        test_dict_theta = theta_list[i]
        # print('\nTest dict theta: {0}'.format(test_dict_theta))
        for item in test_dict:
            # print('Item: {0}'.format(item))
            fs = test_dict[item]
            # print('FS: {0}'.format(fs))
            true_p = list(item)
            # print(type(true_p))
            # print('\nTrue params: {0}'.format(true_p))
            output='results/1d-2epoch/fit-dadi/theta_'+ str(test_dict_theta) + '_'+ str(true_p[0])+'_'+ str(true_p[1])+'_dadi.fitted.params'
            popt = util.infer_demography(fs, func, grids, p0, output,
                     upper_bound, lower_bound)
            # print('POPT: {0}'.format(popt))  #[42.96548938  1.69861494]
            # print(type(popt)) #class 'numpy.ndarray'
            # res.append((list(popt)[0],list(popt)[1],true_p[0],true_p[1]))
            y_true.append(true_p) 
            y_pred.append(list(popt))
            #print('\nRes: {0}'.format(res))
        # print('\nTrue params: {0}'.format(y_true))
        # print('Inferred params: {0}'.format(y_pred))
        
        # Plotting
        param_true, param_pred = util.sort_by_param(y_true, y_pred)
        r2_by_param = util.rfr_r2_score(y_true, y_pred)[1]
        msle_by_param = util.rfr_msle(y_true, y_pred)[1]
        count_param = 1
        for true, pred, r2, msle in zip(param_true, param_pred, 
        r2_by_param, msle_by_param):
            util.plot_by_param(true, pred, r2, msle)
            plt.savefig('test'+str(count_test)+'param'+str(count_param)+'_nlopt_direct'+'.png')
            plt.clf()
            count_param+=1
        count_test+=1
        i+=1

    # # Load the data, example of getting 1 spectrum only
    # data = list(list(enumerate(list_test_dict[0].items()))[0])[1]
    # # get fs
    # fs = list(data)[1]
    # # get params
    # params = list(data)[0]
    # ns = fs.sample_sizes
    # print('True parameters: {0}'.format(params))

    # # Now let's optimize parameters for this model.
    # # The upper_bound and lower_bound lists are for use in optimization.
    # # Occasionally the optimizer will try wacky parameter values. 
    # # We in particular want to exclude values with very long times, 
    # # very small population sizes, or very high migration rates, 
    # # as they will take a long time to evaluate.
    # # Parameters are: (nu1F, nu2B, nu2F, m, Tp, T)
    # upper_bound = [100, 2]
    # lower_bound = [1e-2, 0.1]

    # # This is our initial guess for the parameters, which is somewhat arbitrary.
    # p0 = [2, 0.2]
    # # Make the extrapolating version of our demographic model function.
    # func_ex = dadi.Numerics.make_extrap_log_func(func)
    # # Perturb our parameters before optimization. This does so by taking each
    # # parameter a up to a factor of two up or down.
    # p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound,
    #                             lower_bound=lower_bound)
    # # Do the optimization. By default we assume that theta is a free parameter,
    # # since it's trivial to find given the other parameters. If you want to fix
    # # theta, add a multinom=False to the call.
    # # The maxiter argument restricts how long the optimizer will run. For real 
    # # runs, you will want to set this value higher (at least 10), to encourage
    # # better convergence. You will also want to run optimization several times
    # # using multiple sets of intial parameters, to be confident you've actually
    # # found the true maximum likelihood parameters.
    # # print('Beginning optimization ************************************************')
    # popt = dadi.Inference.optimize_log(p0, fs, func_ex, pts_l, 
    #                                 lower_bound=lower_bound,
    #                                 upper_bound=upper_bound,
    #                                 verbose=0)
    # # The verbose argument controls how often progress of the optimizer should 
    # # be printed. It's useful to keep track of optimization process.
    # # print('Finshed optimization **************************************************')
    # print('Optimized parameters: {0}'.format(popt))