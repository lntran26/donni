from multiprocessing import Pool
import dadi
import math
import copy
import numpy as np
import msprime

def worker_func(args):
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)

def generating_data(params_list, theta_list, func, ns, pts_l,
                        logs=None, bootstrap=False, ncpu=None):
    '''Parallelized version for generating data using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    logs is a list where len(logs) = len(p) and 
    logs[i] = True if p[i] is in log scale. 
    If bootstrap is True, will generate data for bootstrapping purpose.
    Returns a list of dictionaries where each dictionary stores
    a data set for training or testing RFR algorithm. 
    Dictionaries have structure params:fs.
    '''
    if logs != None:
        arg_list =[([10**p[i] if logs[i] else p[i] for i in range(len(logs))],
                        func, ns, pts_l) for p in params_list]
    else:
        arg_list = [(p, func, ns, pts_l) for p in params_list]
    
    with Pool(processes=ncpu) as pool:
            fs_list = pool.map(worker_func, arg_list)
            
    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            if bootstrap:# only run this if want to generate bootstrap data sets
                fs_tostore = (theta*abs(fs)).sample()
                data_dict[params] = [fs_tostore, []]
                for i in range(200):# 200 bootstrap samples for each fs
                    data_dict[params][1].append(fs_tostore.sample())
            else:# generate data for ML training and testing        
                fs.flat[0] = 0
                fs.flat[-1] = 0
                if theta == 1:
                    fs_tostore = fs
                else:
                    fs_tostore = (theta*abs(fs)).sample()
                if fs_tostore.sum()==0:
                    pass
                else:    
                    data_dict[params]=fs_tostore/fs_tostore.sum()                
        list_dicts.append(data_dict)
    return list_dicts

def sort_by_param(y_true, y_pred):
    '''
    Sort the output of model_test into lists of true vs predict 
    values by each param used in the model
    Returns: param_true and param_pred are each a list of lists, each sublist
    contains true or pred values for one param
    '''
    param_true, param_pred = [], []
    n=0
    while n < len(y_true[0]):
        param_list_true, param_list_pred = [], []
        for true, pred in zip(y_true, y_pred):
            param_list_true.append(true[n])
            param_list_pred.append(pred[n])
        param_true.append(param_list_true)
        param_pred.append(param_list_pred)
        n+=1
    return param_true, param_pred

def bootstrap_predictions(trained_model, bootstrap_data):
    '''
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
        pred_dict = {} # dictionary storing prediction results
        for true_p, all_fs in theta_case.items():
            # load elements from bootstrap data
            orig_fs = all_fs[0] # single original fs
            list_bstr_fs = all_fs[1] # list of 200 bootstrapped fs from orig_fs
            # run prediction for the original fs
            orig_fs = orig_fs/orig_fs.sum() # normalize for prediction
            orig_fs_pred = trained_model.predict([orig_fs.flatten()]).flatten()
            # run prediction for fs bootstrapped from original fs
            list_bstr_fs_pred = []
            for bstr_fs in list_bstr_fs:
                bstr_fs = bstr_fs/bstr_fs.sum() # normalize before prediction
                bstr_fs_pred = trained_model.predict([bstr_fs.flatten()]).flatten()
                list_bstr_fs_pred.append(bstr_fs_pred)
            # save all predictions into a dictionary for each theta case
            pred_dict[true_p] = [orig_fs_pred, list_bstr_fs_pred]
        # save all dictionaries into a list for multiple theta cases
        bootstrap_pred.append(pred_dict)
    return bootstrap_pred

def bootstrap_intervals(bootstrap_pred_i, params, percentile=95):
    '''
    Generates a list of intervals sorted by param containing true value, 
    original value, lower bound, and upper bound
    bootstrap_pred_i: a single dict for 1 theta case 
    from the list of dictionaries output from bootstrap_predictions()
    params: list of params used in the model as strings, 
    e.g, ['nu1', 'nu2', 'T', 'm']
    theta_i: the index of desired theta to use from theta_list
    percentile: the desired % confidence interval 
    '''
    keys = list(bootstrap_pred_i.keys()) # get dict keys into a list
    all_intervals_by_param = [ [] for p in params] # list of lists to store results for each param

    for j,key in enumerate(keys):
        preds = bootstrap_pred_i[key] #format:[orig fs pred,[200 bstr_fs pred]]
        #print(f"true: {key}")
        #print(f"orig: {preds[0]}")
        bstr_by_param = sort_by_param_bstr(preds[1]) # sort preds by param
        bounds = (100 - percentile) / 2 / 100 
        offset = int(len(bstr_by_param[0]) * bounds) - 1 # e.g., 5th value is index 4
        for i,cur_p in enumerate(params):
            all_cur_p = bstr_by_param[i] # sorted by current param, e.g., all nu1 preds sorted
            all_cur_p.sort()
            true = key[i]
            orig = preds[0][i]
            low = all_cur_p[offset]
            high = all_cur_p[len(all_cur_p) - offset]
            #print(f'---------{cur_p}---------')
            #print(f'true: {true:.4f}')
            #print(f'orig: {orig:.4f}')
            #print(f'{percentile}% CI: {low:.4f}-{high:.4f}')
            interval = [true, orig, low, high]
            all_intervals_by_param[i].append(interval)
    return all_intervals_by_param
    # return is a list of length len(params), 
    # with inner lists of length len(list_bstr_fs_pred), 
    # with innermost list length 4.
    # E.g., all_intervals_by_param = [nu1, nu2, T, m]
    # where nu1=[list1,...,list200] where list1=[true, orig, lower, upper]

def sort_by_param_bstr(p_sets):
    '''
    Sort by params for bootstrapping workflow.
    '''
    p_sorted = [] # list by param
    for i in range(len(p_sets[0])):
        p = [] # single param list (e.g., nu1)
        for p_set in p_sets:
            p.append(p_set[i])
        p_sorted.append(p)
    return p_sorted

def msprime_two_epoch(s1, p):
    '''Generate msprime demographic model equivalent to
    two_epoch model from dadi with dadi parameters'''
    (nu, T) = p
    dem = msprime.Demography()
    dem.add_population(initial_size=s1*10**nu) # size at present time
    dem.add_population_parameters_change(time=2*s1*T, 
                                        initial_size=s1) # size of ancestral pop

    return dem

def msprime_split_mig(s1, p):
    (nu1, nu2, T, m) = p
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=s1*10**nu1) # pop1 at present time
    dem.add_population(name="B", initial_size=s1*10**nu2) # pop2 at present time
    dem.add_population(name="C", initial_size=s1) # ancestral pop
    dem.add_population_split(time=2*s1*T, derived=["A", "B"], ancestral="C")
    dem.set_symmetric_migration_rate(["A", "B"], m/(2*s1))
    return dem

def msprime_generate_ts(args):
    '''Simulate TS under msprime demography model'''
    (dem, ns, ploidy, seq_l, recomb) = args
    # simuate tree sequences
    return(msprime.sim_ancestry(samples=ns, ploidy=ploidy, demography=dem, 
                                sequence_length=seq_l, 
                                recombination_rate=recomb))

def msprime_generate_data(params_list, dem_list, ns, ploidy, seq_l, 
                            recomb, mut, sample_nodes=None, ncpu=None):
    '''Parallelized version for generating data from msprime 
    using multiprocessing.
    Output format same as generate_data with dadi but FS were simulated
    and summarized from TS data generated under msprime models.'''
    arg_list = [(dem, ns, ploidy, seq_l, recomb) for dem in dem_list]
    with Pool(processes=ncpu) as pool:
        ts_list = pool.map(msprime_generate_ts, arg_list)

    data_dict = {}
    for params, ts in zip(params_list, ts_list):
        # simulate mutation to add variation
        mts = msprime.sim_mutations(ts, rate=mut, discrete_genome=False)
        # Using discrete_genome=False means that the mutation model will 
        # conform to the classic infinite sites assumption, 
        # where each mutation in the simulation occurs at a new site.

        # print statement for troubleshooting only
        # print(f'nu={10**params[0]:.2f}, T={params[1]:.2f}, #SNPs={mts.num_mutations}') # 1d_2epoch
        # print(f'nu1={10**params[0]:.2f}, nu2={10**params[1]:.2f}, T={params[2]:.2f}, m={params[3]:.2f}, #SNPs={mts.num_mutations}') # 2d_splitmig

        # convert mts to afs 
        afs = mts.allele_frequency_spectrum(sample_sets=sample_nodes,
                                        polarised=True, span_normalise=False)
        # polarised=True: generate unfolded/ancestral state known fs
        # span_normalise=False: by default, windowed statistics are divided by the 
        # sequence length, so they are comparable between windows.
        # convert afs to dadi fs object, normalize and save
        fs = dadi.Spectrum(afs)
        if fs.sum() == 0:
            pass
        else:
            data_dict[params] = fs/fs.sum()
    return data_dict

# Test code:
# We protect this test code with this Python idiom. This means the test
# code won't run when we "import util", which is useful for defining
# functions we'll want to use in multiple scripts.
if __name__ == "__main__":
    import time
    import numpy as np
    import random

    theta_list = [1, 1000]
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # testing running time for the generating_data functions (non log)
    test_params = []
    for i in range(100):
        nu = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        params = (round(nu, 2), round(T, 1))
        test_params.append(params)
    
    start = time.time()
    generating_data(test_params, theta_list, func, ns, pts_l)
    print('Parallel execution time to generate data 1D: {0:.2f}s'
    .format(time.time()-start))

    # test log implementation
    test_params_log = []
    while len(test_params_log) < 100:
        nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        test_params_log.append((nu, T))

    logs = [True,False]
    start = time.time()
    list_test_dict = generating_data(test_params_log, theta_list, func, ns, pts_l, logs)
    print('Parallel execution time to generate data 1D log: {0:.2f}s'
    .format(time.time()-start))
    # print(list_test_dict[0].keys())