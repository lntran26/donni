import dadi
import pickle
import numpy as np

# import previously produced benchmarking test sets for 2D-split-migration
test_set_1 = pickle.load(open(
    'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_4','rb'))
test_set_2 = pickle.load(open(
    'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_5','rb'))
test_set_3 = pickle.load(open(
    'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_6','rb'))

# each test set is a list of 60 tuples, each tuple contains
# 3 elements: p_true, fs, and p0

# we want to change test_set_#[20-59][2], which are all
# the p0 starting params each as a list [nu1,nu2,T,m]
def nn_bounds(test_set):
    for i in range(20, 60):
        p = test_set[i][2]
        if p[0] > 10**2:
            p[0] = 10**2
        if p[0] < 10**-2:
            p[0] = 10**-2
        if p[1] > 10**2:
            p[1] = 10**2
        if p[1] < 10**-2:
            p[1] = 10**-2
        if p[2] > 2:
            p[2] = 2
        if p[2] < 0.1:
            p[2] = 0.1
        if p[3] > 10:
            p[3] = 10
        if p[3] < 1:
            p[3] = 1
    return test_set

transform_test_set_1 = nn_bounds(test_set_1)
transform_test_set_2 = nn_bounds(test_set_2)
transform_test_set_3 = nn_bounds(test_set_3)

pickle.dump(transform_test_set_1, open(
    'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_4_bounds',
        'wb'), 2)
pickle.dump(transform_test_set_2, open(
    'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_5_bounds',
        'wb'), 2)
pickle.dump(transform_test_set_3, open(
    'data/2d-splitmig/benchmarking_nn/benchmarking_test_set_6_bounds',
        'wb'), 2)