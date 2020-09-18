from multiprocessing import Pool
import dadi

# A rework of your make_list_dicts function. Note that we pass in the
# demographic function we're working with, to make this function broadly
# applicable.
def training_data(params_list, theta_list, func, ns, pts_l):
    func_ex = dadi.Numerics.make_extrap_func(func)

    # Note previously you were calculating func_ex for each parameter set
    # for each value in theta_list. But the result of func_ex doesn't depend
    # on theta. Because this is the expensive part of the calculation, you were
    # wasting a lot of CPU time. Here I first calculate all the necessary spectra
    # then sample from them later.
    fs_list = [func_ex(p, ns, pts_l) for p in params_list]

    list_dicts = []
    # Note Pythonic way to iterate through lists
    for theta in theta_list:
        data_dict = {}
        # Zip lets us go through these lists together
        for params, fs in zip(params_list, fs_list):
            if theta == 1:
                fs_tostore = fs
            else:
                fs_tostore = (theta*fs).sample()
            data_dict[params] = fs_tostore/fs_tostore.sum()
        list_dicts.append(data_dict)
    return list_dicts

# There are some annoying techincal details working with multiprocessing Pools.
# First, they only work with single-argument functions.
# Second, you can't pass newly-created functions to them. So we need to do
# the make_extrap_func inside it here.
def worker_func(args):
    (p, func, ns, pts_l) = args
    func_ex = dadi.Numerics.make_extrap_func(func)
    return func_ex(p, ns, pts_l)

def training_data_parallel(params_list, theta_list, func, ns, pts_l, ncpu=None):
    # This version uses multiprocessing. If npcu=None, it will use all the CPUs
    # on the machine. Otherwise you can specify a limit.
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
                fs_tostore = (theta*fs).sample()
            data_dict[params] = fs_tostore/fs_tostore.sum()
        list_dicts.append(data_dict)
    return list_dicts

# We protect this test code with this Python idiom. This means the test
# code won't run when we "import util", which is useful for defining
# functions we'll want to use in multiple scripts.
if __name__ == "__main__":
    import time
    import numpy as np

    # Generate test arguments. Note fancy list comprehension usage here...
    # Also, np.linspace is often easier to use than arange.
    params_list = [(nu,T) for nu in 10**np.linspace(-2, 2, 10)
                          for T in np.linspace(0.1,2,10)]
    theta_list = [1, 1000]
    func = dadi.Demographics1D.two_epoch
    ns = [20]
    pts_l = [40, 50, 60]

    start = time.time()
    training_data(params_list, theta_list, func, ns, pts_l)
    print('Serial execution time: {0:.2f}s'.format(time.time()-start))

    start = time.time()
    training_data_parallel(params_list, theta_list, func, ns, pts_l)
    print('Parallel execution time: {0:.2f}s'.format(time.time()-start))