import msprime
import dadi
from multiprocessing import Pool

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

# None parallel version for testing
def msprime_generate_data(params_list, dem_list, ns, ploidy, seq_l, recomb, mut):
    arg_list = [(dem, ns, ploidy, seq_l, recomb, mut) for dem in dem_list]
    fs_list = [msprime_generate_fs(arg) for arg in arg_list]

    data_dict = {}
    for params, fs in zip(params_list, fs_list):
        data_dict[params] = fs
    return data_dict

# Test code: running time for the 1D version (2 epoch)
# We protect this test code with this Python idiom. This means the test
# code won't run when we "import util", which is useful for defining
# functions we'll want to use in multiple scripts.
if __name__ == "__main__":
    import time
    import numpy as np
    import random
    # Generate test arguments.
    params_random = []
    while len(params_random) < 10: 
    # generate random nu and T within the same range as training data range
        nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        # exclude T/nu > 5
        if T/10**nu <= 5:
            ms_params = (nu, T)
            params_random.append(ms_params)
    
    s1 = 5_000
    dem_list_random = [msprime_two_epoch(s1, p) for p in params_random]

    ns = 10
    ploidy = 2
    seq_l = 2_000
    recomb = 1e-3
    mut = 1e-3

    # testing running time for the generating_data function
    start = time.time()
    msprime_generate_data(params_random, dem_list_random, ns, ploidy, seq_l, recomb, mut)
    print('Serial execution time to generate data: {0:.2f}s'
    .format(time.time()-start))
    # Serial execution time to generate data: 88.02s

    start = time.time()
    msprime_generate_data_parallel(params_random, dem_list_random, ns, ploidy, seq_l, recomb, mut)
    print('Parallel execution time to generate data: {0:.2f}s'
    .format(time.time()-start))
    # Parallel execution time to generate data: 29.82s