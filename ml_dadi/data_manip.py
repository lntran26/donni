from multiprocessing import Pool
import dadi
import math
import copy
import numpy as np
import msprime


def worker_func(args):
    '''Helper function for generating_data()
    Help with parallelization with Pool'''
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)


def generating_data(params_list, theta_list, func, ns, pts_l,
                    logs=None, bootstrap=False, norm=True, ncpu=None):
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
        arg_list = [([10**p[i] if logs[i] else p[i] for i in range(len(logs))],
                    func, ns, pts_l) for p in params_list]
    else:
        arg_list = [(p, func, ns, pts_l) for p in params_list]

    with Pool(processes=ncpu) as pool:
        fs_list = pool.map(worker_func, arg_list)

    list_dicts = []
    for theta in theta_list:
        data_dict = {}
        for params, fs in zip(params_list, fs_list):
            if bootstrap:  # only run this if want to generate bootstrap data sets
                fs_tostore = (theta*abs(fs)).sample()
                data_dict[params] = [fs_tostore, []]
                for i in range(200):  # 200 bootstrap samples for each fs
                    data_dict[params][1].append(fs_tostore.sample())
            else:  # generate data for ML training and testing
                fs.flat[0] = 0
                fs.flat[-1] = 0
                if theta == 1:
                    fs_tostore = fs
                else:
                    fs_tostore = (theta*abs(fs)).sample()
                if fs_tostore.sum() == 0:
                    pass
                elif norm:
                    data_dict[params] = fs_tostore/fs_tostore.sum()
                else:
                    data_dict[params] = fs_tostore
        list_dicts.append(data_dict)
    return list_dicts


def worker_func_opt(args):
    '''Helper function for dadi_opt()
    Help with parallelization with Pool'''
    (p, fs, func, pts_l, lb, ub) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return dadi.Inference.opt(p, fs, func_ex, pts_l, lower_bound=lb,
                              upper_bound=ub)


def dadi_opt(p_list, fs_list, func, pts_l, lb, ub, ncpu=None):
    '''Parallelized version for running dadi optimization with multiprocessing.
    If npcu=None, it will use all the CPUs on the machine. 
    Otherwise user can specify a limit.
    '''
    args_list = [(p, fs, func, pts_l, lb, ub)
                 for p, fs in zip(p_list, fs_list)]
    with Pool(processes=ncpu) as pool:
        opt_list = pool.map(worker_func_opt, args_list)
    return opt_list


def converged(top):
    if round(abs(top[0][1][1]-top[1][1][1]), 2) <= 0.05:
        if round(abs(top[0][1][1]-top[2][1][1]), 2) <= 0.05:
            return True
    return False


# bootstrap_predictions() was replaced by ml_models.model_bootstrap()

# bootstrap_intervals() was moved to plotting.py


def msprime_two_epoch(s1, p):
    '''Generate msprime demographic model equivalent to
    two_epoch model from dadi with dadi parameters'''
    (nu, T) = p
    dem = msprime.Demography()
    dem.add_population(initial_size=s1*10**nu)  # size at present time
    dem.add_population_parameters_change(time=2*s1*T,
                                         initial_size=s1)  # size of ancestral pop

    return dem


def msprime_split_mig(s1, p):
    (nu1, nu2, T, m) = p
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=s1*10 **
                       nu1)  # pop1 at present time
    dem.add_population(name="B", initial_size=s1*10 **
                       nu2)  # pop2 at present time
    dem.add_population(name="C", initial_size=s1)  # ancestral pop
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

    logs = [True, False]
    start = time.time()
    list_test_dict = generating_data(
        test_params_log, theta_list, func, ns, pts_l, logs)
    print('Parallel execution time to generate data 1D log: {0:.2f}s'
          .format(time.time()-start))
