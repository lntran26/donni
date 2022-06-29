"""Specifications for each dadi demographic models
supported with mlpr prediction"""
import random
import numpy as np
import dadi
from dadi import Numerics, PhiManip, Integration, Spectrum
import dadinet.portik_models_3d as portik_3d


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
        # nu range: 0.01-100 --> log10 nu range: -2 to 2
        # T range: 0.1-2
        nu = random.random() * 4 - 2
        T = random.random() * 1.9 + 0.1
        params_list.append((nu, T))
    return func, params_list, logs


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
    return func, params_list, logs


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
    return func, params_list, logs


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
    return func, params_list, logs


def _OutOfAfrica(params, ns, pts):
    '''Custom dadi demographic model function not included in API'''
    nuAf, nuB, nuEu0, nuEu, nuAs0, nuAs, \
        mAfB, mAfEu, mAfAs, mEuAs, \
        TAf, TB, TEuAs, Tsum = params  # include Tsum
    xx = Numerics.default_grid(pts)

    phi = PhiManip.phi_1D(xx)
    phi = Integration.one_pop(phi, xx, TAf, nu=nuAf)

    phi = PhiManip.phi_1D_to_2D(xx, phi)
    phi = Integration.two_pops(
        phi, xx, TB, nu1=nuAf, nu2=nuB, m12=mAfB, m21=mAfB)

    phi = PhiManip.phi_2D_to_3D_split_2(xx, phi)

    def nuEu_func(t): return nuEu0 * (nuEu/nuEu0) ** (t/TEuAs)
    def nuAs_func(t): return nuAs0 * (nuAs/nuAs0) ** (t/TEuAs)
    phi = Integration.three_pops(phi, xx, TEuAs, nu1=nuAf, nu2=nuEu_func,
                                 nu3=nuAs_func, m12=mAfEu, m13=mAfAs,
                                 m21=mAfEu, m23=mEuAs, m31=mAfAs, m32=mEuAs)

    fs = Spectrum.from_phi(phi, ns, (xx, xx, xx))
    return fs


def _param_range(param_type):
    ''' Helper function to generate random parameter values
    within biologically realistic range for each type of dem param.
    Input: param_type is a string corresponding to range_dict key'''
    range_dict = {'size': (4, -2),
                  'time': (1.9, 0.1),
                  'mig': (9, 1),
                  's': (0.98, 0.01)}
    a, b = range_dict[param_type]

    return random.random() * a + b


def OutOfAfrica(n_samples):
    '''Specifications for 3D Out of Africa model'''
    # load custom demographic model from helper function
    func = _OutOfAfrica

    # specify param in log scale
    log_options = [True, False]
    rep_time = [6, 8]  # include T_sum
    logs = list(np.repeat(log_options, rep_time))

    # generate params
    params_list = []

    while len(params_list) < n_samples:
        # pick random values in specified range
        p = []
        for _ in range(6):  # get 6 size params
            p.append(_param_range('size'))
        for _ in range(4):  # get 4 migration rate params
            p.append(_param_range('mig'))
        # for _ in range(3):  # get 3 event time params
        #     p.append(_param_range('time'))
        # sample t_sum from time param range, then divide into 3 p's
        t_sum = _param_range('time')
        p += list(np.random.dirichlet(np.ones(3))*t_sum)
        # also include t_sum
        p.append(t_sum)
        # save param values as a tuple
        params_list.append(tuple(p))
    return func, params_list, logs


def OutOfAfrica_no_mig(n_samples):
    '''Specifications for 3D Out of Africa model without migration
    with T sum restricted to be in biologically relevant range'''
    # load custom demographic model from helper function
    func = _OutOfAfrica

    # specify param in log scale
    log_options = [True, False]
    rep_time = [6, 8]  # include T_sum
    logs = list(np.repeat(log_options, rep_time))

    # generate params
    params_list = []

    while len(params_list) < n_samples:
        # pick random values in specified range
        p = []
        for _ in range(6):  # get 6 size params
            p.append(_param_range('size'))
        for _ in range(4):  # get 4 migration rate params
            p.append(0)
        # for _ in range(3):  # get 3 event time params
        #     p.append(_param_range('time'))
        # sample t_sum from time param range, then divide into 3 p's
        t_sum = _param_range('time')
        p += list(np.random.dirichlet(np.ones(3))*t_sum)
        # also include t_sum
        p.append(t_sum)
        # save param values as a tuple
        params_list.append(tuple(p))
    return func, params_list, logs


def split_sym_mig_adjacent_var1(n_samples):
    '''Specifications for a Portik 3D model'''
    # load custom demographic model from helper function
    func = portik_3d.split_sym_mig_adjacent_var1

    # specify param in log scale
    log_options = [True, False]
    rep_time = [4, 5]
    logs = list(np.repeat(log_options, rep_time))

    # generate params
    params_list = []

    while len(params_list) < n_samples:
        # pick random values in specified range
        p = []
        for _ in range(4):  # get 4 size params
            p.append(_param_range('size'))
        for _ in range(3):  # get 3 migration rate params
            p.append(_param_range('mig'))
        for _ in range(2):  # get 2 event time params
            p.append(_param_range('time'))

        # save param values as a tuple
        params_list.append(tuple(p))
    return func, params_list, logs


def split_sym_mig_adjacent_var1_modified(n_samples):
    '''Specifications for a Portik 3D model
    Modified to make all migration values 0'''
    # load custom demographic model from helper function
    func = portik_3d.split_sym_mig_adjacent_var1

    # specify param in log scale
    log_options = [True, False]
    rep_time = [4, 5]
    logs = list(np.repeat(log_options, rep_time))

    # generate params
    params_list = []

    while len(params_list) < n_samples:
        # pick random values in specified range
        p = []
        for _ in range(4):  # get 4 size params
            p.append(_param_range('size'))
        for _ in range(3):  # get 3 migration rate params
            p.append(0)
        for _ in range(2):  # get 2 event time params
            p.append(_param_range('time'))

        # save param values as a tuple
        params_list.append(tuple(p))
    return func, params_list, logs


def null(n_samples):
    """Dummy function to plot dem model with just Tsum"""
    return None, [], [False]


def three_epoch(n_samples):
    '''Specifications for 1D three_epoch model'''
    # designate dadi demographic model
    func = dadi.Demographics1D.three_epoch
    # specify param in log scale
    logs = [True, True, False, False]
    # generate params
    params_list = []
    while len(params_list) < n_samples:
        # pick random values in specified range
        p = []
        for _ in range(2):  # get 2 size params
            p.append(_param_range('size'))
        for _ in range(2):  # get 2 event time params
            p.append(_param_range('time'))
        # save param values as a tuple
        params_list.append(tuple(p))
    return func, params_list, logs


def _three_epoch(params, ns, pts):
    """
    params = (nuB,nuF,TB,TF,Tsum)
    ns = (n1,)

    nuB: Ratio of bottleneck population size to ancient pop size
    nuF: Ratio of contemporary to ancient pop size
    TB: Length of bottleneck (in units of 2*Na generations)
    TF: Time since bottleneck recovery (in units of 2*Na generations)

    n1: Number of samples in resulting Spectrum
    pts: Number of grid points to use in integration.
    """
    nuB, nuF, TB, TF, Tsum = params

    xx = Numerics.default_grid(pts)
    phi = PhiManip.phi_1D(xx)

    phi = Integration.one_pop(phi, xx, TB, nuB)
    phi = Integration.one_pop(phi, xx, TF, nuF)

    fs = Spectrum.from_phi(phi, ns, (xx,))
    return fs


def three_epoch_restricted(n_samples):
    '''Specifications for 1D three_epoch model
    with T_sum range restricted to 0.1 to 2'''
    # designate dadi demographic model
    func = _three_epoch
    # specify param in log scale
    logs = [True, True, False, False, False]
    # generate params
    params_list = []
    while len(params_list) < n_samples:
        # pick random values in specified range
        p = []
        for _ in range(2):  # get 2 size params
            p.append(_param_range('size'))
        t_sum = _param_range('time')
        p += list(np.random.dirichlet(np.ones(2))*t_sum)
        # also include t_sum
        p.append(t_sum)
        # save param values as a tuple
        params_list.append(tuple(p))
    return func, params_list, logs
