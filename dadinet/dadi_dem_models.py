"""Specifications for each dadi demographic models
supported with mlpr prediction"""
import random
import dadi


def two_epoch(n_samples):
    '''Specifications for 1D two_epoch model'''
    # designate dadi demographic model
    func = dadi.Demographics1D.two_epoch
    # specify param in log scale
    logs = [True, False]
    # generate params
    params_list = []
    while len(params_list) < n_samples:
        # pick random values in specified range
        # nu range: 0.01-100; T range: 0.1-2
        nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        params_list.append((nu, T))
    return params_list, func, logs


def growth(n_samples):
    '''Specifications for 1D growth model'''
    # designate dadi demographic model
    func = dadi.Demographics1D.growth
    # specify param in log scale
    logs = [True, False]
    # generate params
    params_list = []
    while len(params_list) < n_samples:
        # pick random values in specified range
        # nu range: 0.01-100; T range: 0.1-2
        nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        params_list.append((nu, T))
    return params_list, func, logs


def split_mig(n_samples):
    '''Specifications for 2D split migration model'''
    # designate dadi demographic model
    func = dadi.Demographics2D.split_mig
    # specify param in log scale
    logs = [True, True, False, False]
    # generate params
    params_list = []
    while len(params_list) < n_samples:
        # pick random values in specified range
        nu1 = random.random() * 4 - 2
        nu2 = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        m = random.random() * 9 + 1
        params_list.append((nu1, nu2, T, m))
    return params_list, func, logs


def IM(n_samples):
    '''Specifications for 2D IM model'''
    # designate dadi demographic model
    func = dadi.Demographics2D.IM
    # specify param in log scale
    logs = [False, True, True, False, False, False]
    # generate params
    params_list = []
    while len(params_list) < n_samples:
        # pick random values in specified range
        s = random.random() * 0.98 + 0.01
        nu1 = random.random() * 4 - 2
        nu2 = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        m12 = random.random() * 9 + 1
        m21 = random.random() * 9 + 1
        params_list.append((s, nu1, nu2, T, m12, m21))
    return params_list, func, logs
