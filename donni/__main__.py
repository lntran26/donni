"""Command-line interface setup for donni"""
import argparse
import pickle
import re
import sys
import os
import dadi
from scipy.stats._distn_infrastructure import rv_frozen as distribution
from donni.dadi_dem_models import get_model, get_param_values
from donni.generate_data import generate_fs, get_hyperparam_tune_dict,\
    fs_quality_check
from donni.train import prep_data, tune, report,\
    get_best_specs, train, get_cv_score
from donni.infer import infer, prep_fs_for_ml, irods_download, irods_cleanup, project_fs
from donni.plot import plot


# run_ methods for importing methods from other modules
def run_generate_data(args):
    '''Method to generate data given inputs from the
    generate_data subcommand'''

    if args.save_individual_fs:
        # check if outdir is provided
        if args.outdir is None:
            sys.exit('donni generate_data: error: '
                     'the following arguments are required:'
                     ' --outdir when using --save_individual_fs')

    # get dem function and params specifications for model
    dadi_func, param_names, logs = get_model(args.model,
                                             args.model_file, args.folded)
    # get demographic param values
    params_list = get_param_values(param_names, args.n_samples, args.seed)

    if not args.generate_tune_hyperparam_only:
        # generate data
        data, qual = generate_fs(dadi_func, params_list, logs,
                                 args.theta, args.sample_sizes, args.grids,
                                 args.non_normalize, args.no_sampling,
                                 args.folded, args.bootstrap, args.n_bstr,
                                 args.n_cpu)

        # output fs quality check results
        if not args.no_fs_qual_check:
            fs_quality_check(qual, args.outfile,
                             params_list, param_names, logs)

        # save data as a dictionary or as individual files
        # (in addition to saving as a single file)
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

        # save data dict as one pickled file (default)
        pickle.dump(data, open(args.outfile, 'wb'))

    # generate and save hyperparams dict for tuning
    if args.generate_tune_hyperparam_only or args.generate_tune_hyperparam:
        tune_dict = get_hyperparam_tune_dict(args.sample_sizes)
        pickle.dump(tune_dict, open(f'{args.hyperparam_outfile}', 'wb'))

        # output text file for details of hyper param tune dict
        with open(f'{args.hyperparam_outfile}.txt', 'w') as fh:
            for hyperparam, option in tune_dict.items():
                if isinstance(option, distribution):
                    distribution_name = vars(
                        option.dist)["name"]
                    distribution_vals = vars(option)["args"]
                    fh.write(f'{hyperparam}: {distribution_name}, '
                             f'{distribution_vals}\n')
                else:
                    fh.write(f'{hyperparam}: {option}\n')


def _process_param_dict_tune(args):
    """
    Helper method for processing input into dict of
    hyperparams used for tuning and/or training
    """
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

    return param_dict


def _process_train_param_dict_list(args, param_dict, y_label):
    """
    Helper method for processing input into list of dicts of
    hyperparams for training of mapie MLPRs when hyperparam file
    is not provided or when hyperparams are specified individually
    from command line
    """
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

    return train_param_dict_list


def run_train(args):
    '''Method to train MLPR given inputs from the train subcommand'''

    # Load training data
    data = pickle.load(open(args.data_file, 'rb'))
    # parse data into input and corresponding labels
    X_input, y_label = prep_data(data, mapie=args.multioutput)
    # get hyperparam dictionary for tuning or training
    param_dict = _process_param_dict_tune(args)
    # make dir to save trained MLPs
    try:
        os.makedirs(args.mlpr_dir)
    except FileExistsError:
        pass
    # Tuning, which generate train_param_dict_list used for training
    if args.tune or args.tune_only:
        all_results = tune(X_input, y_label, param_dict,
                           args.max_iter, args.eta, args.cv, args.multioutput)
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
        # get hyperparam of best mlpr from tuning results, which
        # is a list of 1 if sklearn, list of multiple if mapie
        train_param_dict_list, scores = get_best_specs(all_results)
        # output train_param_dict_list will be input into train() for training
        pickle.dump(train_param_dict_list, open(
            f'{args.mlpr_dir}/tuned_hyperparam_dict_list', 'wb'))
        # print best scores after outputing the mlpr model
        with open(f'{args.mlpr_dir}/tune_results_brief.txt', 'a') as fh:
            for i, (spec, score) in enumerate(zip(train_param_dict_list,
                                                  scores)):
                fh.write(f'CV score of best MLPR for param {i+1}: {score}\n')
                fh.write(f'Spec of best MLPR for param {i+1}: {spec}\n')
    else:
        # Train directly without tuning first.
        # Generate train_param_dict_list with either hyperparam for
        # sklearn MLPR or hyperparam_list for mapie MLPR.
        # If neither is provided, hyperparam will be generated using the
        # default options for each hyperparam from the command line.
        train_param_dict_list = _process_train_param_dict_list(
            args, param_dict, y_label)

    # Training with train_param_dict_list
    if not args.tune_only:
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
    filename_list = sorted(os.listdir(args.mlpr_dir))
    if "param_all_predictor" in filename_list:
        mapie = False  # this is the sklearn case
        mlpr = pickle.load(
            open(os.path.join(args.mlpr_dir, "param_all_predictor"), 'rb'))
        mlpr_list = [mlpr]
    else:
        for filename in filename_list:
            if filename.startswith("param") and filename.endswith("predictor"):
                mlpr = pickle.load(
                    open(os.path.join(args.mlpr_dir, filename), 'rb'))
                mlpr_list.append(mlpr)

    # need to get logs to de-log prediction
    _, param_names, logs = get_model(args.model, args.model_file,
                                     args.folded)  # now included misid
    # this way of getting logs misses one log value for misid,
    # which is currently added only in after running generate_data
    # module helper function
    # check hot fix in plot.py line #228
    # and line #208 below in run_predict()

    return mlpr_list, mapie, logs, param_names


def run_infer(args):
    '''Method to get prediction given inputs from the
    predict subcommand'''

    # open input FS from file
    fs = dadi.Spectrum.from_file(args.input_fs)
    
    if args.mlpr_dir != None:
        # load trained MLPRs and demographic model logs; TODO: remove for cloud support
        mlpr_list, mapie, logs, param_names = _load_trained_mlpr(args)
    else:
        fs = project_fs(fs)
        ss = fs.sample_sizes
        args.folded = fs.folded
        username, password = args.download_mlpr
        args.mlpr_dir = irods_download(username, password, args.model, ss, args.folded)
        # load trained MLPRs and demographic model logs; TODO: remove for cloud support
        mlpr_list, mapie, logs, param_names = _load_trained_mlpr(args)
    # load func
    func, _, _= get_model(args.model, args.model_file, args.folded)
    pis_list = sorted(args.pis)
    # infer params using input FS
    pred, theta, pis = infer(mlpr_list, func, fs, logs, mapie=mapie, pis=pis_list)
    # write output
    if args.output_prefix:
        output_stream = open(args.output_prefix, 'w')
    else:
        output_stream = sys.stdout
    pred.append(theta)
    pi_names = []
    for i, pi in enumerate(pis_list):
        for j, param in enumerate(param_names):
            pi_names.append(param + "_lb_" + str(pi))
            pi_names.append(param + "_ub_" + str(pi))
            pred.append(pis[j][i][0])
            pred.append(pis[j][i][1])
    print_names = param_names + ["theta"] + pi_names
    # print parameter names
    print("# ", end="", file=output_stream)
    print(*print_names, sep='\t', file=output_stream)
    # print prediction
    print(*pred, sep='\t', file=output_stream)
    print(file=output_stream)  # newline
    # print readable intervals
    print(f"{'# CIs: ':<10}", end="", file=output_stream)
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
    mlpr_list, mapie, logs, param_names = _load_trained_mlpr(args)

    # load test fs set
    test_dict = pickle.load(open(args.test_dict, 'rb'))

    # prepare fs in test_dict for ml prediction:
    # check that fs is normalized and masked entries set to 0
    prep_test_dict = {}
    for params_key in test_dict:
        prep_test_dict[params_key] = prep_fs_for_ml(test_dict[params_key])

    # parse test dict into test FS and corresponding labels
    X_test, y_test = prep_data(prep_test_dict, mapie=mapie)

    # load train fs set
    train_dict = pickle.load(open(args.train_dict, 'rb'))

    # parse data into input and corresponding labels
    X_input, y_label = prep_data(train_dict, mapie=True)

    # make result dir to save results
    try:
        os.makedirs(args.results_dir)
    except FileExistsError:
        pass

    # plot results
    plot_prefix = os.path.join(args.results_dir, args.plot_prefix)
    plot(mlpr_list, X_test, y_test, X_input, y_label, plot_prefix,
         args.mlpr_dir, logs, mapie=mapie, coverage=args.coverage,
         theta=args.theta, params=param_names)


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


def donni_parser():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Demography Optimization via Neural Network Inference',
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
                                      type=str, required=True,
                                      help="Path to save generated data and\
                                         associated quality check file")
    generate_data_parser.add_argument('--save_individual_fs',
                                      action='store_true',
                                      help="Save individual FS as a file\
                                        instead of together in one dictionary")
    generate_data_parser.add_argument('--outdir',
                                      type=str,
                                      help="Dir to save individual FS")
    generate_data_parser.add_argument('--grids', type=_pos_int,
                                      nargs=3, help='Sizes of grids',
                                      default=None)
    generate_data_parser.add_argument('--theta', type=_pos_int,
                                      help="Factor to multiply FS with",
                                      default=1)
    generate_data_parser.add_argument('--seed', type=_pos_int,
                                      help="Seed for reproducibility")
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
                                      help="Generate hyperparam spec for\
                                         tuning")
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
                              help='Maximum iterations')
    train_parser.add_argument('--eta', type=_int_2, default=3,
                              help='Halving factor')
    train_parser.add_argument('--cv', type=_int_2, default=5,
                              help='k-fold cross validation')

    # optional input for a pickled dict file instead of setting params manually
    # with flags
    # if provided will be prioritized over input flags
    train_parser.add_argument("--hyperparam", type=str, help="Path to pickled\
                                dict of MLPR hyperparam")
    # flags for specifying mlpr hyperparams if not providing dict
    train_parser.add_argument("--hyperparam_list", type=str, help="Path to\
                                pickled list of dict of hyperparam")
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
                              help='Options: identity, logistic, tanh, relu',
                              default=['relu'])
    train_parser.add_argument('--solver', nargs='*', metavar='NAME',
                              choices=['lbfgs', 'adam'], default=['adam'],
                              help='Options: lbfgs, adam')  # excluded sgd
    train_parser.add_argument('--alpha', nargs='*', type=float,
                              help='L2 penalty regularization param')
    train_parser.add_argument('--tol', nargs='*', type=float,
                              help='Tolerance for optimization')
    train_parser.add_argument('--early_stopping', nargs='*', type=bool,
                              help='Whether to use early stopping')
    train_parser.add_argument('--beta1', nargs='*', type=float,
                              help='Exp decay rate of 1st moment in adam')
    train_parser.add_argument('--beta2', nargs='*', type=float,
                              help='Exp decay rate of 2nd moment in adam')
    train_parser.add_argument('--n_iter_no_change', nargs='*', type=int,
                              help='Max epochs to not meet tol improvement')

    # subcommand for infer
    infer_parser = subparsers.add_parser(
        "infer", help='Use trained MLPR to infer demographic parameters\
            from frequency spectra data')
    infer_parser.set_defaults(func=run_infer)
    # need to handle dir for multiple models for mapie
    # single dir for sklearn models
    infer_parser.add_argument("--input_fs", type=str, required=True,
                                help="Path to FS file for generating\
                                     inference")
    infer_parser.add_argument('--model', type=str,
                                required=True,
                                help="Name of dadi demographic model")
    # Arg for downloading MLPRs
    if '--mlpr_dir' not in sys.argv:
        download_req = True
    else:
        download_req = False
    infer_parser.add_argument('--download_mlpr', nargs=2,
                                default=[], action="store",
                                required=download_req,
                                help="Pass in your username and password for the CyVerse Data Store to download MLPR models. Required if user did not make their own MLPRs for inference.")
    # Arg for users that made their own MLPRs
    if '--download_mlpr' not in sys.argv:
        path_req = True
    else:
        path_req = False
    infer_parser.add_argument("--mlpr_dir", type=str, required=path_req,
                                help="Path to saved, trained MLPR(s). Required if user is not downloading MLPRs for inference.")
    infer_parser.add_argument('--folded', action="store_true",
                                      help="Whether to fold FS")
    # optional
    infer_parser.add_argument("--pis", type=_pos_int,
                                nargs='+', default=[95],
                                help="Optional list of values for\
                                    inference intervals,\
                                    e.g., [80 90 95]; default [95]")
    infer_parser.add_argument('--model_file', type=str,
                                help="Name of file containing custom dadi\
                                     demographic model(s)",)
    infer_parser.add_argument("--output_prefix", type=str,
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
    plot_parser.add_argument("--train_dict", type=str, required=True,
                             help="Path to train data dictionary file")
    plot_parser.add_argument("--results_dir", type=str, required=True,
                             help="Directory to save output plots")
    plot_parser.add_argument("--plot_prefix", type=str, required=True,
                             help="Prefix for plot filenames")
    plot_parser.add_argument('--model', type=str,
                             required=True,
                             help="Name of dadi demographic model")

    # optional
    plot_parser.add_argument('--model_file', type=str,
                             help="Name of file containing custom dadi\
                                 demographic model(s)")
    plot_parser.add_argument('--coverage', action='store_true', default=False,
                             help="Generate coverage plot (used with mapie)")
    plot_parser.add_argument('--theta', type=_pos_int,
                             help="Theta used to generate test_dict for\
                                 labeling plots",
                             default=None)
    plot_parser.add_argument('--folded', action="store_true",
                             help="Specify if the test FS is folded")

    return parser


def main(arg_list=None):
    """Main program"""

    parser = donni_parser()
    args = parser.parse_args(arg_list)
    args.func(args)
