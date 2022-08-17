import dadi
import dadi.DFE as DFE
import sys
import os
import importlib
from inspect import getmembers, isfunction
import random
import numpy as np

duplicated_models = ["snm", "bottlegrowth"]
duplicated_sele_models = [
    "IM",
    "IM_pre",
    "IM_pre_single_gamma",
    "IM_single_gamma",
    "split_asym_mig",
    "split_asym_mig_single_gamma",
    "split_mig",
    "three_epoch",
    "two_epoch",
    "split_mig_single_gamma",
]
oned_models = [m[0] for m in getmembers(dadi.Demographics1D, isfunction)]
twod_models = [m[0] for m in getmembers(dadi.Demographics2D, isfunction)]
sele_models = [m[0] for m in getmembers(DFE.DemogSelModels, isfunction)]

for m in duplicated_models:
    oned_models.remove(m)
for m in duplicated_models:
    twod_models.remove(m)
for m in duplicated_sele_models:
    sele_models.remove(m)

def get_model(model_name, n_samples, model_file=None):
    """
    Description:
        Obtains a demographic model, its parameters.

    Arguments:
        model_name str: Name of the demographic model.
        model_file str: Path and Name of the file containing customized models.

    Returns:
        func function: Demographic model for modeling.
        params list: List of parameters.
        logs list: List of True/False to denote if a parameter is in log-scale
    """
    model_name0 = model_name
    if model_file != None:
        # If the user has the model folder in their PATH
        try:
            func = getattr(importlib.import_module(model_file), model_name)
        # If the user does not have the model folder in their PATH we add it
        # This currently can mess with the User's PATH while running dadi-cli
        except:
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

    # Generate parameter and log values.
    # Think about possibly breaking this off
    # into a new function to fetch parameter and log values.
    try:
        param_names = func.__param_names__
        param_lists = [[] for ele in range(n_samples)]
        logs = ["nu" in ele for ele in param_names]
        for i in range(n_samples):
            T_fraction_list = np.random.dirichlet(np.ones(sum(["T" in ele for ele in param_names])))
            T_fraction_index = 0
            t_sum = _param_range("T")
            for param_name in param_names:
                # Can probably build upon later for DFE parameters
                if "nu" in param_name:
                    param_type = "nu"
                    param_lists[i].append(_param_range(param_type))
                elif "T" in param_name:
                    param_type = "T"
                    param_lists[i].append(t_sum*T_fraction_list[T_fraction_index])
                    T_fraction_index+=1
                else:
                    param_type = param_name[:1]
                    param_lists[i].append(_param_range(param_type))

    except:
        raise ValueError(
            f'Demographic model needs a .__param_names__ attribute!\nAdd one by adding the line '
            + model_name0
            + '.__param_name__ = [LIST_OF_PARAMS]\nReplacing LIST_OF_PARAMS with the names of the parameters as strings.\n'
            + 'For parameter naming:\n\tSize changes should start with "nu".\n\tTimes should start with "T".'
            + '\n\tMigration rates should start with "m".\n\tPopulation splits should start with "s".'
        )

    return func, param_lists, logs

def _param_range(param_type):
    ''' Helper function to generate random parameter values
    within biologically realistic range for each type of dem param.
    Input: param_type is a string corresponding to range_dict key'''
    range_dict = {"nu": (4, -2),
                  "T": (1.9, 0.1),
                  "m": (9, 1),
                  "s": (0.98, 0.01)}
    a, b = range_dict[param_type]

    return random.random() * a + b

