"""Command-line interface setup for dadi-ml"""
import argparse
import pickle
import re
import sys
import os
from inspect import getmembers, isfunction
import numpy as np
import dadi
from scipy.stats._distn_infrastructure import rv_frozen as distribution
from dadinet.dadi_dem_models import get_model
from dadinet.generate_data import generate_fs, get_hyperparam_tune_dict
from dadinet.train import prep_data, tune, report,\
    get_best_specs, train, get_cv_score
from dadinet.predict import predict, prep_fs_for_ml
from dadinet.plot import plot


# run_ methods for importing methods from other modules
def run_generate_data(args):
    '''Method to generate data given inputs from the
    generate_data subcommand'''

    # get dem function and params specifications for model
    dadi_func, params_list, logs, param_names = get_model(
        args.model, args.n_samples, args.model_file)

    if not args.generate_tune_hyperparam_only:
        # generate data
        data = generate_fs(dadi_func, params_list, logs,
                           args.theta, args.sample_sizes, args.grids,
                           args.non_normalize, args.no_sampling, args.folded,
                           args.bootstrap, args.n_bstr, args.n_cpu)

        # save data as a dictionary or as individual files
        if args.save_individual_fs:
            # make dir to save individual fs and true params to
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)
            # process data dict to individual fs and save
            # index in fs file name matches index in true_log_params list
            true_log_params = list(data.keys())
            for i, p in enumerate(true_log_params):
                fs = data[p]
                fs.tofile(f"{args.outdir}/fs_{i:03d}")
            pickle.dump(true_log_params, open(
                f'{args.outdir}/true_log_params', 'wb'))
        else:
            # save data dict as one pickled file
            pickle.dump(data, open(args.outfile, 'wb'))
            # check fs quality
            # to do: add this option to save individual fs
            # need to change output location accordingly
            if not args.no_fs_qual_check:
                keys = list(data.keys())
                neg_fs = 0
                nan_fs = 0
                inf_fs = 0
                for key in keys:
                    fs = data[key]
                    if not args.bootstrap:
                        if np.any(fs < 0):
                            neg_fs += 1
                        if np.any(np.isnan(fs)):
                            nan_fs += 1
                        if np.any(np.isposinf(fs)):
                            inf_fs += 1
                with open(f'{args.outfile}_quality.txt', 'a') as fh:
                    fh.write(f'Quality check for {args.outfile}:\n')
                    fh.write(
                        f'Number of FS with at least one negative entry: {neg_fs}\n')
                    fh.write(
                        f'Number of FS with at least one NaN entry: {nan_fs}\n')
                    fh.write(
                        f'Number of FS with at least one pos inf entry: {inf_fs}\n')

    if args.generate_tune_hyperparam_only or args.generate_tune_hyperparam:
        tune_dict = get_hyperparam_tune_dict(args.sample_sizes)
        pickle.dump(tune_dict, open(f'{args.hyperparam_outfile}', 'wb'))

        # output text file for details of hyper param tune dict
        with open(f'{args.hyperparam_outfile}.txt', 'w') as fh:
            for hyperparam in tune_dict:
                if isinstance(tune_dict[hyperparam], distribution):
                    distribution_name = vars(
                        tune_dict[hyperparam].dist)["name"]
                    distribution_vals = vars(tune_dict[hyperparam])["args"]
                    fh.write(
                        f'{hyperparam}:{distribution_name}, {distribution_vals}\n')
                else:
                    fh.write(f'{hyperparam}: {tune_dict[hyperparam]}\n')


def run_train(args):
    '''Method to train MLPR given inputs from the
    train subcommand'''
    # TO DO: update complex if-else trees to handle tune v. train
    # and using hyperparam vs hyperparam_list more efficiently

    # Load training data
    data = pickle.load(open(args.data_file, 'rb'))
    # parse data into input and corresponding labels
    X_input, y_label = prep_data(data, mapie=args.multioutput)

    # process input from command line into a dict of hyperparams
    if args.hyperparam is not None:
        param_dict = pickle.load(open(args.hyperparam, 'rb'))
    else:
        # if hyperparam dict is not provided then use individual default
        # hyperparam arg to create one
        param_dict = {}
        excluded_args = ['data_file', 'mlpr_dir', 'multioutput', 'tune',
                         'max_iter', 'subcommand', 'func', 'hyperparam',
                         'eta', 'cv', 'hyperparam_list', 'tune_only',
                         'training_score']
        for arg in vars(args):
            if arg not in excluded_args and getattr(args, arg) is not None:
                param_dict[arg] = getattr(args, arg)
    # # for debugging
    # print(f'param_dict: {param_dict}')
    if args.tune or args.tune_only:
        # run tuning using input param_dict
        all_results = tune(X_input, y_label, param_dict,
                           args.max_iter, args.eta, args.cv)
        # output full tuning result file
        pickle.dump(all_results, open(
            f'{args.mlpr_dir}/tune_results_full', 'wb'))
        # output abbreviated printed result
        with open(f'{args.mlpr_dir}/tune_results_brief.txt', 'wt') as fh:
            for i, model in enumerate(all_results):
                fh.write(f'MLPR for param {i+1}:'.center(50, '*'))
                fh.write('\n')
                for j, _ in enumerate(model):
                    band = all_results[i][j]
                    str1 = f'\nBand {j+1}: Fitting {band.n_candidates_} '
                    str2 = f'candidates for {band.n_resources_} iterations\n'
                    fh.write(str1 + str2)
                    report(band.cv_results_, fh)
        # get train hyperparam from best mlpr from tuning
        # list of 1 if sklearn, list of multiple if mapie
        train_param_dict_list, scores = get_best_specs(all_results)
        # output train_param_dict_list, which is the list of tuned hyperparam
        # dicts that will be input into train() for training
        pickle.dump(train_param_dict_list, open(
            f'{args.mlpr_dir}/tuned_hyperparam_dict_list', 'wb'))
        # print best scores after outputing the mlpr model
        with open(f'{args.mlpr_dir}/tune_results_brief.txt', 'a') as fh:
            for i, (spec, score) in enumerate(zip(train_param_dict_list,
                                                  scores)):
                fh.write(f'CV score of best MLPR for param {i+1}: {score}\n')
                fh.write(f'Spec of best MLPR for param {i+1}: {spec}\n')
        # print("Finish tuning\n")

    else:
        # alternatively, train directly without tuning first.
        # This will train with either hyperparam or hyperparam_list if provided.
        # If neither is provided, hyperparam will be generated using the default
        # options for each hyperparam from the command line.

        # prioritizing reading hyperparam_list over hyperparam if provided
        if args.hyperparam_list is not None and args.multioutput:
            # hyperparam_list only works for mapie, not sklearn multioutput
            # args.multioutput is true if using mapie
            train_param_dict_list = pickle.load(
                open(args.hyperparam_list, 'rb'))
        else:
            # only process hyperparam input into a list
            # if hyperparam_list is not provided
            train_param_dict = {}
            for key, value in param_dict.items():
                # handling potentially incorrect input for tune instead of train
                if isinstance(value, list):  # if input is a list
                    # get only the first value in list for each hyperparam
                    train_param_dict[key] = value[0]
                elif isinstance(value, distribution):
                    pass  # ignore if input is a scipy distribution
                else:  # get expected input value as a single input
                    train_param_dict[key] = value
            # append one hyperparam dict for each demographic param/ MLPR
            # by making the length of hyperparam dict equal length of y_label
            train_param_dict_list = []
            for _ in range(len(y_label)):
                train_param_dict_list.append(train_param_dict)
    # # for debugging
    # print(f'train_param_dict_list: {train_param_dict_list}\n')

    if not args.tune_only:
        # print("Start training\n")
        # train with best hyperparams from tuning or with input if not tuning
        trained = train(X_input, y_label, train_param_dict_list,
                        mapie=args.multioutput)

        # save trained mlpr(s)
        for i, mlpr in enumerate(trained):
            index = f'{i+1:02d}' if args.multioutput else 'all'
            pickle.dump(mlpr, open(
                f'{args.mlpr_dir}/param_{index}_predictor', 'wb'))

        # output cv score of trained mlpr on training set
        if args.training_score:
            with open(f'{args.mlpr_dir}/training_score.txt', 'wt') as fh:
                get_cv_score(trained, X_input, y_label, fh, cv=args.cv)


def _load_trained_mlpr(args):
    """Helper method to read in trained MLPR models for predict and plot"""

    mlpr_list = []
    mapie = True
    for filename in sorted(os.listdir(args.mlpr_dir)):
        if filename.startswith("param") and filename.endswith("predictor"):
            mlpr = pickle.load(
                open(os.path.join(args.mlpr_dir, filename), 'rb'))
            mlpr_list.append(mlpr)
            if filename == "param_all_predictor":
                mapie = False  # this is the sklearn case
                break
        else:
            continue
    # need to get logs to de-log prediction
    func, _, logs, param_names = get_model(args.model, 0, args.model_file)
    # this way of getting logs misses one log value for misid,
    # which is currently added only in after running generate_data
    # module helper function
    # check hot fix in plot.py line #228
    # and line #208 below in run_predict()

    return mlpr_list, mapie, logs, param_names


def run_predict(args):
    '''Method to get prediction given inputs from the
    predict subcommand'''

    # open input FS from file
    fs = dadi.Spectrum.from_file(args.input_fs)
    # load trained MLPRs and demographic model logs
    mlpr_list, mapie, logs, param_names = _load_trained_mlpr(args)
    pis_list = sorted(args.pis)
    # misid case
    if not fs.folded:
        logs.append(False)
        param_names.append("misid")
    # infer params using input FS
    pred, pis = predict(mlpr_list, fs, logs, mapie=mapie, pis=pis_list)
    # write output
    if args.output_prefix:
        output_stream = open(args.output_prefix, 'w')
    else:
        output_stream = sys.stdout

    pi_names = []
    for i, pi in enumerate(pis_list):
        for j, param in enumerate(param_names):
            pi_names.append(param + "_lb_" + str(pi))
            pi_names.append(param + "_ub_" + str(pi))
            pred.append(pis[j][i][0])
            pred.append(pis[j][i][1])
    print_names = param_names + pi_names
    # print parameter names
    print("# ", end="", file=output_stream)
    print(*print_names, sep='\t', file=output_stream)
    # print prediction
    print(*pred, sep='\t', file=output_stream)
    print(file=output_stream)  # newline
    # print readable confidence intervals
    print(f"{'# PIs: ':<10}", end="", file=output_stream)
    for pi in pis_list:
        print(f"|----------{pi}----------|", end='\t', file=output_stream)
    print(file=output_stream)
    for i, param in enumerate(param_names):
        print(f"{'# ' + param + ': ':<10}", end="", file=output_stream)
        for pi in pis[i]:
            print(f"[{pi[0]:10.6f}, {pi[1]:10.6f}]",
                  end="\t", file=output_stream)
        print(file=output_stream)
    if args.output_prefix:
        output_stream.close()


def run_plot(args):
    '''Method to plot outputs'''

    # load trained MLPRs and demographic model logs
    mlpr_list, mapie, logs, _ = _load_trained_mlpr(args)

    # load test fs set
    test_dict = pickle.load(open(args.test_dict, 'rb'))

    # prepare fs in test_dict for ml prediction:
    # check that fs is normalized and masked entries set to 0
    prep_test_dict = {}
    for params_key in test_dict:
        prep_test_dict[params_key] = prep_fs_for_ml(test_dict[params_key])

    # parse data into test FS and corresponding labels
    X_test, y_test = prep_data(prep_test_dict, mapie=mapie)

    # Get param names
    # Want to do one sample so that
    # get_model can test custom model file?
    _, _, _, param_names = get_model(args.model, 0, args.model_file)
    # Check if misid is a parameter to be added
    if not args.folded:
        param_names += ['misid']

    # plot results
    plot(mlpr_list, X_test, y_test, args.results_prefix, logs, mapie=mapie,
         coverage=args.coverage, theta=args.theta, params=param_names)


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

    try:
        for single_tup in re.split(' ', input_str):
            return tuple(map(int, single_tup.split(',')))
    except Exception as error:
        raise argparse.ArgumentTypeError(
            "Hidden layers must be divided by commas," +
            " e.g. 'h1,h1 h2,h2,h2'") from error


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

    generate_data_parser.add_argument('--model', type=str,
                                      required=True,
                                      help="Name of dadi demographic model",)
    generate_data_parser.add_argument('--model_file', type=str,
                                      help="Name of file containing custom dadi\
                                         demographic model(s)",)
    # --model will dictate params_list, func, logs, and param_names
    generate_data_parser.add_argument('--n_samples', type=_pos_int,
                                      required=True,
                                      help="How many FS to generate",)
    generate_data_parser.add_argument('--sample_sizes', type=_pos_int,
                                      nargs='+', required=True,
                                      help="Sample sizes of populations",)
    generate_data_parser.add_argument('--outfile',
                                      type=str, default="training_fs",
                                      help="Path to save generated data")
    generate_data_parser.add_argument('--save_individual_fs',
                                      action='store_true',
                                      help="Save individual FS as a file\
                                        instead of together in one dictionary")
    generate_data_parser.add_argument('--outdir',
                                      type=str,
                                      help="Dir to save individual FS")
    generate_data_parser.add_argument('--grids', type=_pos_int,
                                      nargs=3, help='Sizes of grids',
                                      default=[40, 50, 60])
    generate_data_parser.add_argument('--theta', type=_pos_int,
                                      help="Factor to multiply FS with",
                                      default=1)
    generate_data_parser.add_argument('--non_normalize', action='store_false',
                                      help="Don't normalize FS")
    generate_data_parser.add_argument('--no_sampling', action='store_false',
                                      help="Don't sample FS when theta > 1")
    generate_data_parser.add_argument('--folded', action="store_true",
                                      help="Whether to fold FS")
    generate_data_parser.add_argument('--bootstrap', action='store_true',
                                      help="Whether to generate bootstrap\
                                           FS data")
    generate_data_parser.add_argument('--n_bstr', type=_pos_int,
                                      help="Number of bootstrap FS to generate\
                                         for each FS (if bootstrap)",
                                      default=200)
    generate_data_parser.add_argument('--n_cpu', type=_pos_int,
                                      help="Number of CPUs to use")
    generate_data_parser.add_argument('--no_fs_qual_check',
                                      action='store_true',
                                      help="Turn off default FS quality check")
    generate_data_parser.add_argument('--generate_tune_hyperparam',
                                      action='store_true',
                                      help="Generate hyperparam spec for tuning")
    generate_data_parser.add_argument('--generate_tune_hyperparam_only',
                                      action='store_true',
                                      help="Generate hyperparam spec for tuning\
                                      only without generating any data")
    generate_data_parser.add_argument('--hyperparam_outfile',
                                      type=str,
                                      default="param_dict_tune",
                                      help="Path to save hyperparam dict\
                                         for tuning")

    # subcommand for train
    train_parser = subparsers.add_parser(
        "train", help='Train MLPR with frequency spectra data')
    train_parser.set_defaults(func=run_train)
    train_parser.add_argument("--data_file", type=str, required=True,
                              help="Path to input training data")
    train_parser.add_argument("--mlpr_dir", type=str, required=True,
                              help="Path to save output trained MLPR(s)")
    train_parser.add_argument("--multioutput", action='store_false',
                              help="Train multioutput sklearn MLPR instead of\
                                  default mapie single-output MLPRs")
    train_parser.add_argument("--tune", action='store_true',
                              help="Whether to try a range of hyperparameters\
                                   to find the best performing MLPRs")
    train_parser.add_argument("--tune_only", action='store_true',
                              help="When use will not refit to train\
                                 on the full data set after tuning")
    train_parser.add_argument("--training_score", action='store_true',
                              help="When use will output training score")
    # hyperband tuning params
    train_parser.add_argument('--max_iter', type=_int_2, default=243,
                              help='maximum iterations')
    train_parser.add_argument('--eta', type=_int_2, default=3,
                              help='halving factor')
    train_parser.add_argument('--cv', type=_int_2, default=5,
                              help='k-fold cross validation')

    # optional input for a pickled dict file instead of setting params manually
    # with flags
    # if provided will be prioritized over input flags
    train_parser.add_argument("--hyperparam", type=str,
                              help="Path to pickled dict of MLPR hyperparam")
    # flags for specifying mlpr hyperparams if not providing dict
    train_parser.add_argument("--hyperparam_list", type=str,
                              help="Path to pickled list of dict of hyperparam")
    # currently: hyperparam expects a single dict, which will be duplicated
    # in the mapie case by the number of params in the demographic model
    # hyperparam_list allows different hyperparam dicts for each demographic
    # param, which is outputed by tuning mapie MLPRs

    train_parser.add_argument('--hidden_layer_sizes',
                              metavar='TUPLE OF POSITIVE INT', nargs='*',
                              #   action='store', dest='hidden_layer_sizes',
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
    predict_parser.add_argument("--input_fs", type=str, required=True,
                                help="Path to FS file for generating\
                                     prediction")
    predict_parser.add_argument('--model', type=str,
                                required=True,
                                help="Name of dadi demographic model")
    predict_parser.add_argument("--mlpr_dir", type=str, required=True,
                                help="Path to saved, trained MLPR(s)")

    # optional
    predict_parser.add_argument("--pis", type=_pos_int,
                                nargs='+', default=[95],
                                help="Optional list of values for\
                                    prediction intervals,\
                                    e.g., [80 90 95]; default [95]")
    predict_parser.add_argument('--model_file', type=str,
                                help="Name of file containing custom dadi demographic model(s)",)
    predict_parser.add_argument("--output_prefix", type=str,
                                help="Optional output file to write out results\
                                   (default stdout)")

    # subcommand for plot
    plot_parser = subparsers.add_parser(
        "plot", help='Plot prediction results and statistics')
    plot_parser.set_defaults(func=run_plot)

    plot_parser.add_argument("--mlpr_dir", type=str, required=True,
                             help="Path to trained MLPR(s)")
    plot_parser.add_argument("--test_dict", type=str, required=True,
                             help="Path to test data dictionary file")
    plot_parser.add_argument("--results_prefix", type=str, required=True,
                             help="Path to save output plots")
    plot_parser.add_argument('--model', type=str,
                             required=True,
                             help="Name of dadi demographic model")
    plot_parser.add_argument('--model_file', type=str,
                             help="Name of file containing custom dadi demographic model(s)",)

    # optional
    plot_parser.add_argument('--coverage', action='store_true', default=False,
                             help="Generate coverage plot (used with mapie)")
    plot_parser.add_argument('--theta', type=_pos_int,
                             help="Theta used to generate test_dict",
                             default=None)
    plot_parser.add_argument('--folded', action="store_true",
                             help="Specify if the test FS is folded")

    return parser


def main(arg_list=None):
    """Main program"""

    parser = dadi_ml_parser()
    args = parser.parse_args(arg_list)
    args.func(args)
