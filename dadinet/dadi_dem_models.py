"""Module for getting demographic models and associated
parameter range values"""
import sys
import os
import importlib
from inspect import getmembers, isfunction
import random
import numpy as np
import dadi


duplicated_models = ["snm", "bottlegrowth"]

oned_models = [m[0] for m in getmembers(dadi.Demographics1D, isfunction)]
twod_models = [m[0] for m in getmembers(dadi.Demographics2D, isfunction)]

for m in duplicated_models:
    oned_models.remove(m)
for m in duplicated_models:
    twod_models.remove(m)


def get_model(model_name, model_file=None, folded=False):
    """
    Description:
        Obtains a demographic model and its parameter info: name and log.
    Arguments:
        model_name str: Name of the demographic model.
        model_file str: Path and Name of the file containing customized models.
        folded bool: whether to include misid param for unfolded FS
    Returns:
        func function: Demographic model for modeling.
        param_names list: List of demographic parameter names.
        logs list: List of True/False to denote if a parameter is in log-scale
    """
    model_name0 = model_name
    if model_file is not None:
        # If the user has the model folder in their PATH
        try:
            func = getattr(importlib.import_module(model_file), model_name)
        # If the user does not have the model folder in their PATH we add it
        # This currently can mess with the User's PATH while running dadi-cli
        except ModuleNotFoundError:
            model_file = os.path.abspath(model_file)
            model_path = os.path.dirname(model_file)
            model_file = os.path.basename(model_file)
            model_file = os.path.splitext(model_file)[0]
            sys.path.append(model_path)
            func = getattr(importlib.import_module(model_file), model_name)
    elif model_name in oned_models:
        func = getattr(dadi.Demographics1D, model_name)
    elif model_name in twod_models:
        func = getattr(dadi.Demographics2D, model_name)
    else:
        raise ValueError(f"Cannot find model: {model_name}.")
    try:
        param_names = list(func.__param_names__)
        # avoid using param_names = func.__param_names__ as to not
        # accidentally modify a function attribute later
        logs = ["nu" in ele for ele in param_names]
        if not folded:
            param_names.append("misid")
            logs.append(False)
    except Exception as error:
        raise ValueError(
            'Demographic model needs a .__param_names__ attribute!\n'
            'Add one by adding the line '
            f'{model_name0}.__param_names__ = [LIST_OF_PARAMS]\n'
            'Replacing LIST_OF_PARAMS with the names of the parameters'
            ' as strings.\n'
            'For parameter naming:\n\tSize changes should start with "nu".'
            '\n\tTimes should start with "T".'
            '\n\tMigration rates should start with "m".'
            '\n\tPopulation splits should start with "s".'
        ) from error

    return func, param_names, logs


def get_param_values(param_names, n_samples,
                     seed=None, seed_offset=0):
    """
    Generate a list of randomly selected demographic parameter values.
    Input:
        param_names list: List of demographic parameter names.
        n_samples int: number of unique param sets
        seed int: seed value to generate the same set of param values
        seed_offset int: offset the seed selection range to ensure
            different seeds selected for generating training vs. test data
    Output:
        params_list list: List of length n_samples of parameter values.
    """
    # random seed settings
    seed_len = n_samples * len(param_names) * 2
    if seed is not None:
        random.seed(seed)  # this seed control random.sample()
        seed_list = random.sample(
            range(seed_offset, seed_len + seed_offset), seed_len)
    else:
        seed_list = [None] * seed_len
    # generate param values
    params_list = []
    s_idx = 0
    for _ in range(n_samples):
        p = []
        # special handling for T param with Tsum restriction
        n_T = sum([name.startswith("T") for name in param_names])
        np.random.seed(seed_list[s_idx])
        T_fraction_list = np.random.dirichlet(np.ones(n_T))
        T_fraction_index = 0
        t_sum = _param_range("T", seed_list[s_idx])
        s_idx += 1
        for name in param_names:
            if name.startswith("T"):
                p_val = t_sum * T_fraction_list[T_fraction_index]
                T_fraction_index += 1
            elif name.startswith("nu"):
                p_val = _param_range("nu", seed_list[s_idx])
            elif name == "misid":
                p_val = _param_range("misid", seed_list[s_idx])
            else:
                p_val = _param_range(name[0], seed_list[s_idx])
            p.append(p_val)
            s_idx += 1
        params_list.append(tuple(p))

    return params_list


def print_built_in_models():
    """
    Description:
        Outputs built-in models in dadi.
    """
    print("Built-in 1D demographic models:")
    for model in oned_models:
        print(f"- {model}")
    print()

    print("Built-in 2D demographic models:")
    for model in twod_models:
        print(f"- {model}")
    print()


def _param_range(param_type, seed=None):
    ''' Helper function to generate random parameter values
    within biologically realistic range for each type of dem param.
    Input: param_type is a string corresponding to range_dict key'''
    range_dict = {"nu": (4, -2, 1),
                  "T": (1.9, 0.1, 1),
                  "m": (9, 1, 1),
                  "s": (0.98, 0.01, 1),
                  "F": (1, 0, 1),
                  "misid": (1, 0, 4)}
    a, b, c = range_dict[param_type]
    random.seed(seed)
    return (random.random() * a + b) / c
