"""Command-line interface setup for dadi-ml"""
import argparse
import pickle
from inspect import getmembers, isfunction
import dadinet.dadi_dem_models as models
from dadinet.train import *

# get demographic model names and functions from dadi_dem_models
model_name, model_func = zip(*getmembers(models, isfunction))
dem_dict = dict(zip(model_name, model_func))


# run_ methods for importing methods from other modules
def run_generate_data(args):
    '''Method to generate data given inputs from the
    generate_data subcommand'''

    from dadinet.generate_data import generate_fs
    # get dem function from input model name
    func = dem_dict[args.model]

    # get params specifications for model
    dadi_func, params_list, logs = func(args.n_samples)

    # generate data
    data = generate_fs(dadi_func, params_list, logs,
                       args.theta, args.sample_sizes,
                       args.grids, args.normalize, args.sampling,
                       args.bootstrap, args.n_bstr, args.n_cpu)

    # save data to output dir
    pickle.dump(data, open(args.outfile, 'wb'), 2)


def run_train(args):
    '''Method to train MLPR given inputs from the
    train subcommand'''

    # Load training data
    data = pickle.load(open(args.data_file, 'rb'))
    # parse data into input and corresponding labels
    X_input, y_label = prep_data(data, mapie=args.mapie)
    # process input from command line into a dictionary of params
    param_dict = {}
    for arg in vars(args):
        if arg not in ['data_file', 'mlpr_dir', 'mapie', 'tune', 'max_iter',
                       'eta', 'cv'] and getattr(args, arg) is not None:
            param_dict[arg] = getattr(args, arg)

    if args.tune:
        # further process hyperparams that are floats: alpha, tol, beta1, beta2
        # if len=2 implement distribution, else leave as a list of discrete val
        for arg in ['alpha', 'tol', 'beta1', 'beta2']:
            arg_value = param_dict[arg]
            if len(arg_value) == 2:
                update_arg_value = make_distribution(
                    arg_value[0], arg_value[1])
                param_dict[arg] = update_arg_value
        # run tuning
        all_results = tune(X_input, y_label, param_dict,
                           args.max_iter, args.eta, args.cv)
        # output full tuning result file
        pickle.dump(all_results, open(
            f'{mlpr_dir}/tune_results_full', 'wb'), 2)
        # output abbreviated printed result
        with open(f'{mlpr_dir}/tune_results_brief.txt', 'wt') as fh:
            for i, model in enumerate(all_results):
                fh.write(f'MLPR for param {i+1}:')
                for j, each_band in enumerate(model):
                    fh.write(f'Band {j+1}:')
                    report(all_results[i][j].cv_results_, fh.name)
        # get train hyperparam from best mlpr from tuning
        train_param_dict, scores = get_best_specs(all_results)
        # print best scores after outputing the mlpr model
        with open(f'{mlpr_dir}/tune_results_brief.txt', 'a') as fh:
            for i, score in enumerate(scores):
                fh.write(f'\nCV score of best MLPR for param {i+1}: {score}')

    else:  # train directly without tuning first
        # get only the first value in list for each hyperparam
        train_param_dict = {}
        for key, value in param_dict.items():
            train_param_dict[key] = value[0]

    # train with best hyperparams from tuning or with input if not tuning
    trained = train(X_input, y_label, train_param_dict, mapie=args.mapie)

    # save trained mlpr(s)
    for i, mlpr in trained:
        index = i+1 if args.mapie else 'all'
        pickle.dump(mlpr, open(
            f'{mlpr_dir}/param_{index}_predictor', 'wb'), 2)
    # output cv score of trained mlpr on training set
    with open(f'{mlpr_dir}/training_score.txt', 'wt') as fh:
        get_cv_score(trained, X_input, y_label, fh.name, cv=args.cv)


def run_predict(args):
    '''Method to get prediction given inputs from the
    predict subcommand'''

    from dadinet.predict import predict
    ...


def run_plot(args):
    '''Method to plot outputs'''

    from dadinet.plot import plot
    ...


# def run_tune(args):
#     '''Method to use hyperband for parameter tuning'''

#     from dadinet.tune import tune
#     ...


# helper methods for custom type checks and parsing
def _pos_int(input_int):
    """
    Check positive integer
    """

    if int(input_int) < 0:
        raise argparse.ArgumentTypeError(f"{input_int} is not a positive int")
    return int(input_int)


def _tuple_of_pos_int(input_str):
    """
    Custom type check for hidden layer sizes input from command line.
    Convert comma-separated string input to tuples of pos int.
    """

    single_tup_int = map(int, input_str.split(','))
    if any(layer_size < 1 for layer_size in single_tup_int):
        raise argparse.ArgumentTypeError(
            f"invalid tuple_of_pos_int value: '{input_str}'")
    return tuple(single_tup_int)


def _int_2(input_int):
    """
    Check if input is an integer >= 2."""
    if int(input_int) < 2:
        raise argparse.ArgumentTypeError("input must be >= 2")
    return int(input_int)


def dadi_ml_parser():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Machine learning applications for dadi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # subcommand for generate_data
    generate_data_parser = subparsers.add_parser(
        'generate_data',
        help='Generate frequency spectra datasets')
    generate_data_parser.set_defaults(func=run_generate_data)

    generate_data_parser.add_argument('--model', type=str, choices=model_name,
                                      required=True,
                                      help="Name of dadi demographic model",)
    # --model will dictate params_list, func, and logs
    generate_data_parser.add_argument('--n_samples', type=_pos_int,
                                      required=True,
                                      help="How many FS to generate",)
    generate_data_parser.add_argument('--sample_sizes', type=_pos_int,
                                      nargs='+', required=True,
                                      help="Sample sizes of populations",)
    generate_data_parser.add_argument('--outfile',
                                      type=str, required=True,
                                      help="Path to save generated data")
    generate_data_parser.add_argument('--grids', type=_pos_int,
                                      nargs=3, help='Sizes of grids',
                                      default=[40, 50, 60])
    generate_data_parser.add_argument('--theta', type=_pos_int,
                                      help="Factor to multiply FS with",
                                      default=1)
    generate_data_parser.add_argument('--normalize', action='store_true',
                                      help="Whether to normalize FS when\
                                           theta > 1", default=True)
    generate_data_parser.add_argument('--sampling', action='store_true',
                                      help="Whether to sample FS when\
                                          theta > 1", default=True)
    generate_data_parser.add_argument('--bootstrap', action='store_true',
                                      help="Whether to generate bootstrap\
                                           FS data")
    generate_data_parser.add_argument('--n_bstr', type=_pos_int,
                                      help="Number of bootstrap FS to generate\
                                         for each FS (if bootstrap)",
                                      default=200)
    generate_data_parser.add_argument('--n_cpu', type=_pos_int,
                                      help="Number of CPUs to use")

    # subcommand for train
    train_parser = subparsers.add_parser(
        "train", help='Train MLPR with frequency spectra data')
    train_parser.set_defaults(func=run_train)
    train_parser.add_argument("--data_file", type=str, required=True,
                              help="Path to input training data")
    train_parser.add_argument("--mlpr_dir", type=str, required=True,
                              help="Path to save output trained MLPR(s)")
    train_parser.add_argument("--mapie", action='store_true', default=True,
                              help="Whether to train multioutput sklearn MLPR\
                                   or mapie single-output MLPRs")
    train_parser.add_argument("--tune", action='store_true',
                              help="Whether to try a range of hyperparameters\
                                   to find the best performing MLPRs")
    # hyperband tuning params
    train_parser.add_argument('--max_iter', type=_int_2,
                              help='maximum iterations, default None=243')
    train_parser.add_argument('--eta', type=_int_2,
                              help='halving factor, default None=3')
    train_parser.add_argument('--cv', type=_int_2,
                              help='k-fold cross validation, default None=5')

    # train_parser.add_argument("--hyperparam", type=str, required=True,
    #                           help="Path to dictionary of MLPR hyperparam")
    # flags for specifying different mlpr hyperparams
    train_parser.add_argument('--hidden_layer_sizes',
                              metavar='TUPLE OF POSITIVE INT', nargs='*',
                              type=_tuple_of_pos_int, default=[(64,)],
                              help='Use commas to separate layers')
    train_parser.add_argument('--activation', nargs='*', metavar='NAME',
                              choices=['identity', 'logistic', 'tanh', 'relu'],
                              help='options: identity, logistic, tanh, relu',
                              default=['relu'])
    train_parser.add_argument('--solver', nargs='*', metavar='NAME',
                              choices=['lbfgs', 'adam'], default=['adam'],
                              help='options: lbfgs, adam')  # excluded sgd
    train_parser.add_argument('--alpha', nargs='*', type=float,
                              help='L2 penalty regularization param')
    train_parser.add_argument('--tol', nargs='*', type=float,
                              help='tolerance for optimization')
    train_parser.add_argument('--early_stopping', nargs='*', type=bool,
                              help='Whether to use early stopping')
    train_parser.add_argument('--beta1', nargs='*', type=float,
                              help='Exp decay rate of 1st moment in adam')
    train_parser.add_argument('--beta2', nargs='*', type=float,
                              help='Exp decay rate of 2nd moment in adam')
    train_parser.add_argument('--n_iter_no_change', nargs='*', type=int,
                              help='Max epochs to not meet tol improvement')

    # subcommand for predict
    predict_parser = subparsers.add_parser(
        "predict", help='Use trained MLPR to predict demographic parameters\
            from frequency spectra data')
    predict_parser.set_defaults(func=run_predict)
    # need to handle dir for multiple models for mapie
    # single dir for sklearn models
    predict_parser.add_argument("model_dir")
    # predict_parser.add_argument("output_dir")
    # predict_parser.add_argument("text_dir")

    # need to have stat flags for getting scores and prediction intervals
    # predict_parser.add_argument("--evaluate", dest='reference_dir')

    # subcommand for plot
    plot_parser = subparsers.add_parser(
        "plot", help='Plot prediction results and statistics')
    plot_parser.set_defaults(func=run_plot)
    plot_parser.add_argument("data_dir")

    # # subcommand for tune
    # tune_parser = subparsers.add_parser(
    #     "tune", help='MLPR hyperparam tuning with hyperband')
    # tune_parser.set_defaults(func=run_tune)
    # tune_parser.add_argument("data_dir")

    return parser


def main(arg_list=None):
    """Main program"""
    parser = dadi_ml_parser()
    args = parser.parse_args(arg_list)
    args.func(args)
