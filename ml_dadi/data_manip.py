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
            fs.flat[0] = 0
            fs.flat[-1] = 0
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

# ! log-scale params, define logs
# repeat of above functions, except some params have been log-transformed
# logs is a list where len(logs) = len(p) and logs[i] = True if p[i] is 
# in log scale 
def worker_func_log(args):
    (p, func, ns, pts_l, logs) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex([10**p[i] if logs[i] else p[i] for i in range(len(logs))],
                   ns, pts_l)
                   
# ! log-scale params, define logs
# takes extra argument logs, as defined above
def generating_data_parallel_log(params_list, theta_list, 
                                func, ns, pts_l, logs, ncpu=None):
    '''Parallelized version for generating_data using multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    logs is a list specifying the indices of params that are in log scale
    Returns a list of dictionaries where each dictionary stores
    a data set for training or testing RFR algorithm. Dictionaries
    have structure params:fs.
    '''
    arg_list = [(p, func, ns, pts_l, logs) for p in params_list]
    with Pool(processes=ncpu) as pool:
            fs_list = pool.map(worker_func_log, arg_list)

    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            fs.flat[0] = 0
            fs.flat[-1] = 0
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

def generating_data_log_bootstraps(params_list, theta_list, 
                                func, ns, pts_l, logs, ncpu=None):
    '''
    Generate data with bootstrapping. Creates a dictionary that maps the params
    used to generate the fs (key) to a list of length 2 (value). The first
    item in the list is the original fs data set. The second item in the list 
    is a list (length 200) of the bootstrap data obtained by sampling from the 
    original fs.
    Note: does not take theta=1 since no sampling can be done on smooth data.
    '''
    arg_list = [(p, func, ns, pts_l, logs) for p in params_list]
    with Pool(processes=ncpu) as pool:
            fs_list = pool.map(worker_func_log, arg_list)

    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            fs_tostore = (theta*abs(fs)).sample()
            data_dict[params] = [fs_tostore, []]
            for i in range(200):
                data_dict[params][1].append(fs_tostore.sample())       
        list_dicts.append(data_dict)
    return list_dicts

def bootstrap_intervals(datafile, params, theta_i, percentile=95):
    '''
    pass a list of params as argument, e.g., [nu1, nu2, T, m]
    '''
    bs_results = pickle.load(open(datafile, 'rb'))
    pred_theta = bs_results[theta_i] # dict of preds for fs scaled by theta with index [100, 1000, 10000]
    keys = list(pred_theta.keys())
    all_intervals_by_param = [[], [], [], []]
    for j,key in enumerate(keys):
        preds = pred_theta[key] # list of predictions from [orig fs pred, [200 boot fs pred]]
        #print(f"true: {key}")
        #print(f"orig: {preds[0]}")
        bs_by_param = sort_by_param(preds[1])
        bounds = (100 - percentile) / 2 / 100 
        offset = int(len(bs_by_param[0]) * bounds) - 1 # e.g., 5th value is index 4
        for i,cur_p in enumerate(params):
            all_cur_p = bs_by_param[i] # sorted by current param, e.g., all nu1 preds sorted
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
    # list length 4, inner list length 200, inner list length 4 [nu1, nu2, T, m]
    # where nu1 = [list 1, list 2 ...] where list 1 = [true, orig, lower, upper]

def log_transform_data(list_data_dict, num_list):
    """
    log transform specified params in input datasets
    input data_dict has structure params:fs 
    num_list is a number list e.g. [0,1] specifying 
    the ith parameter to be log transformed
    """
    transformed_list_data_dict = []
    for data_dict in list_data_dict:
        transformed_data_dict = {}
        params=data_dict.keys()
        fs = [data_dict[params] for params in data_dict]
        transformed_params = []
        # loop through p in params, which is a list of param tuples
        # for each tuple p, copy value if [i] is not in num_list,
        # log transform value if [i] is in num_list
        # store into tuple, then into transformed_param list
        # need to convert tuple to a list because tuples
        # are immutable
        for p in params:
            p = list(p)
            for n in num_list:
                p[n]=math.log10(p[n])
                # p[n]=1/p[n] 
            # convert list back to tuple for making dictionary keys
            transformed_params.append(tuple(p))
        transformed_data_dict=dict(zip(transformed_params, fs))
        transformed_list_data_dict.append(transformed_data_dict)
    return transformed_list_data_dict

def un_log_transform_predict(y_predict, num_list):
    # transformed_predict = y_predict.copy()
    transformed_predict = copy.deepcopy(y_predict)
    for p in transformed_predict:
        for n in num_list:
            p[n] = 10**p[n]
    return transformed_predict

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

def msprime_two_epoch(s1, p):
    (nu, T) = p
    dem = msprime.Demography()
    dem.add_population(initial_size=s1*10**nu) # size at present time
    dem.add_population_parameters_change(time=2*s1*T, 
                                        initial_size=s1) # size of ancestral pop

    return dem

def msprime_generate_fs(args):
    (dem, ns, ploidy, seq_l, recomb, mut) = args
    # simuate tree sequences
    ts = msprime.sim_ancestry(samples=ns, ploidy=ploidy, demography=dem, 
                                sequence_length=seq_l, 
                                recombination_rate=recomb)
    # simulate mutation to add variation
    mts = msprime.sim_mutations(ts, rate=mut, discrete_genome=False)
    # Using discrete_genome=False means that the mutation model will conform 
    # to the classic infinite sites assumption, 
    # where each mutation in the simulation occurs at a new site.
    
    # convert tree sequence to allele frequency spectrum
    afs = mts.allele_frequency_spectrum(polarised=True, span_normalise=False)
    # polarised=True: generate unfolded/ancestral state known fs
    # span_normalise=False: by default, windowed statistics are divided by the 
    # sequence length, so they are comparable between windows.
    
    # convert to dadi fs object
    fs = dadi.Spectrum(afs)
    if fs.sum() == 0:
        pass
    else:
        fs_tostore = fs/fs.sum()
    return fs_tostore

def msprime_generate_data_parallel(params_list, dem_list, ns, ploidy, seq_l, recomb, mut, ncpu=None):
    arg_list = [(dem, ns, ploidy, seq_l, recomb, mut) for dem in dem_list]
    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(msprime_generate_fs, arg_list)

    data_dict = {}
    for params, fs in zip(params_list, fs_list):
        data_dict[params] = fs
    return data_dict

# Test code:
# We protect this test code with this Python idiom. This means the test
# code won't run when we "import util", which is useful for defining
# functions we'll want to use in multiple scripts.
if __name__ == "__main__":
    import time
    import numpy as np
    import random
    train_params = [(nu,T) for nu in 10**np.linspace(-2, 2, 10)
                          for T in np.linspace(0.1,2,10)]

    test_params = []
    for i in range(100):
        nu = 10 ** (random.random() * 4 - 2)
        T = random.random() * 1.9 + 0.1
        params = (round(nu, 2), round(T, 1))
        test_params.append(params)
    
    theta_list = [1, 1000]
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    # testing running time for the generating_data_parallel function
    start = time.time()
    generating_data_parallel(train_params, theta_list, func, ns, pts_l)
    print('Parallel execution time to generate data 1D: {0:.2f}s'
    .format(time.time()-start))