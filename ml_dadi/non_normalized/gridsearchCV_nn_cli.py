#!/usr/bin/env python
"""
Author : linhtran <linhtran@localhost>
Date   : 2021-10-23
Purpose: Perform GridsearchCV to search for MLPR hyperparameter
"""

import argparse
import os
import sys
import pickle
import re
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

# todo: requirement.txt to install scikit-learn
# might not need dadi as we just use the fs as array of numbers


def _tuple_of_int(input_str_or_list):
    """
    Helper function to parse hidden layer sizes input from command line.
    Convert comma-separated inputs from command line to tuples."""
    try:
        for single_tup in re.split(' ', input_str_or_list):
            return tuple(map(int, single_tup.split(',')))
    except:
        raise argparse.ArgumentTypeError(
            "Hidden layers must be divided by space, e.g. 'h1,h1 h2,h2,h2'")


def _check_legitimate_activation(activation_name):
    if activation_name not in ['identity', 'logistic', 'tanh', 'relu']:
        raise argparse.ArgumentTypeError(
            f'{activation_name} is not a valid activation function')
    return activation_name


def _check_legitimate_solver(solver_name):
    if solver_name not in ['lbfgs', 'sgd', 'adam']:
        raise argparse.ArgumentTypeError(
            f'{solver_name} is not a valid optimizer')
    return solver_name


def _check_legitimate_learning_rate(learning_rate):
    if learning_rate not in ['constant', 'invscaling', 'adaptive']:
        raise argparse.ArgumentTypeError(
            f'{learning_rate} is not a valid learning rate')
    return learning_rate

# --------------------------------------------------


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Perform GridsearchCV to search for MLP hyperparameter',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data',
                        metavar='pickle data file',
                        help='Pickled data file to use for hyperparam search')

    parser.add_argument('-hls',
                        '--hidden_layer_sizes',
                        nargs='*',
                        metavar='tuple of int',
                        type=_tuple_of_int,
                        action='store',
                        dest='hidden_layer_sizes',
                        help='Tuple(s) of hidden layer sizes',
                        default=[(100,)])

    parser.add_argument('-a',
                        '--activation',
                        nargs='*',
                        metavar=str,
                        type=_check_legitimate_activation,
                        action='store',
                        dest='activation',
                        help='Name(s) of activation function(s)',
                        default=['relu'])

    parser.add_argument('-s',
                        '--solver',
                        nargs='*',
                        metavar=str,
                        type=_check_legitimate_solver,
                        action='store',
                        dest='solver',
                        help='Name(s) of solver(s)',
                        default=['adam'])

    parser.add_argument('-lr',
                        '--learning_rate',
                        nargs='*',
                        metavar=str,
                        type=_check_legitimate_learning_rate,
                        action='store',
                        dest='learning_rate',
                        help='Name(s) of learning_rate(s) for sgd',
                        default=None)

    parser.add_argument('-mi',
                        '--max_iter',
                        nargs='*',
                        type=int,
                        action='store',
                        dest='max_iter',
                        help='Maximum number of iterations',
                        default=[500])

    parser.add_argument('-l2',
                        '--alpha',
                        nargs='*',
                        type=float,
                        action='store',
                        dest='alpha',
                        help='L2 penalty regularization param',
                        default=None)

    parser.add_argument('-es',
                        '--early_stopping',
                        metavar='boolean',
                        type=bool,
                        action='store',
                        dest='early_stopping',
                        nargs='*',
                        help='Whether to use early stopping',
                        default=None)

    parser.add_argument('-t',
                        '--tol',
                        type=float,
                        action='store',
                        dest='tol',
                        help='tolerance for optimization with early stopping',
                        default=None)

    parser.add_argument('-n',
                        '--n_iter_no_change',
                        type=int,
                        action='store',
                        dest='n_iter_no_change',
                        help='Maximum n epochs to not meet tol improvement',
                        default=None)

    parser.add_argument('-v',
                        '--verbose',
                        type=int,
                        help='Level of GridsearchCV Verbose',
                        default=4)

    # parser.add_argument('-o',
    #                     '--outfile',
    #                     help='Output filename',
    #                     metavar='FILE',
    #                     type=argparse.FileType('wt'),
    #                     default=sys.stdout)

    # parser.add_argument('-e',
    #                     '--errfile',
    #                     help='Error filename',
    #                     metavar='FILE',
    #                     type=argparse.FileType('wt'),
    #                     default=sys.stderr)

    return parser.parse_args()


# --------------------------------------------------
def model_search(model, train_dict, param_dict, verbose=4):
    '''Use GridSearchCV to search for the best hyperparameters
    for ML models

    Input:
    model: ML algorthims such as MLPR
    train_dict: dictionary of training data used for the tuning
    param_dict: dictionary of different model hyperparameters to be tuned
    n_top: int, number of top results to print out
    verbose: int, level of details of the run to output to stdout

    Output: results dictionary
    '''

    # load training data from train_dict
    train_features = [train_dict[params].data.flatten()
                      for params in train_dict]
    train_labels = [params for params in train_dict]

    # perform grid search using selected model, data, and params
    cv = GridSearchCV(model, param_dict, n_jobs=-1, verbose=verbose)
    cv.fit(train_features, train_labels)

    return cv.cv_results_


# --------------------------------------------------
def main():
    """Main program"""

    args = get_args()

    # Load training data
    # check if the file exists:
    if os.path.isfile(args.data):
        train_dict = pickle.load(open(args.data, 'rb'))
    else:
        print(f'Error: {args.data}: File not found', file=sys.stderr)

    # Specify the ML models to be optimized
    mlpr = MLPRegressor()

    # # Reroute all output to outfiles if specified
    # if args.outfile is not sys.stdout:
    #     sys.stdout = args.outfile
    # if args.errfile is not sys.stderr:
    #     sys.stderr = args.errfile

    # process input from command line into a dictionary of params
    param_dict = {}
    for arg in vars(args):
        if arg not in ['data', 'outfile', 'errfile',
                       'verbose'] and getattr(args, arg) != None:
            param_dict[arg] = getattr(args, arg)
            print(arg, ':', getattr(args, arg))#, file=args.outfile)

    # get the total number of models being tested from param_dict
    n_models = 1
    for hyperparam in param_dict:
        n_models *= len(param_dict[hyperparam])

    # call model_search to run gridsearch and store results
    results = model_search(mlpr, train_dict, param_dict, args.verbose)

    # todo: process results to print out certain things
    for i in range(1, n_models + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('\n', f'Model with rank: {i}')#, file=args.outfile)
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))#,
                  #file=args.outfile)
            print(f"Parameters: {results['params'][candidate]}")#,
                  #file=args.outfile)


# --------------------------------------------------
if __name__ == '__main__':
    main()
