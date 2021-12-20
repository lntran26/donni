"""Command-line interface setup for dadi-ml"""
import argparse
import pickle
from inspect import getmembers, isfunction
import dadi_dem_models

# get demographic model names and functions from dadi_dem_models
model_name, model_func = zip(*getmembers(dadi_dem_models, isfunction))
dem_dict = dict(zip(model_name, model_func))


def run_generate_data(args):
    '''Method to generate data given inputs from the
    generate_data subcommand'''
    from generate_data import generate_fs
    # get dem function from input model name
    func = dem_dict[args.model]
    # # for debugging only
    # print(func.__name__)

    # get params specifications for model
    dadi_func, params_list, logs = func(args.n_samples)
    # # for debugging only
    # print(dadi_func.__name__)
    # print(params_list)
    # print(logs)

    # generate data
    data = generate_fs(dadi_func, params_list, logs,
                       args.theta, args.sample_sizes,
                       args.grids, args.sampling,
                       args.normalize, args.bootstrap,
                       args.n_bstr, args.n_cpu)

    # # for debugging only
    # print(len(data))
    # print(data[0])

    # save data to output dir
    pickle.dump(data, open(args.outdir, 'wb'), 2)


def run_train(args):    
    '''Method to train MLPR given inputs from the
    train subcommand'''
    from train import train
    ...


def run_predict(args):
    '''Method to get prediction given inputs from the
    predict subcommand'''
    from predict import predict
    ...


def dadi_ml_parser():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Machine learning applications for dadi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(required=True)

    # subcommand for generate_data
    generate_data_parser = subparsers.add_parser(
        'generate_data',
        help='Generate frequency spectra datasets')
    generate_data_parser.set_defaults(func=run_generate_data)

    generate_data_parser.add_argument('--model', type=str,
                                      choices=model_name,
                                      required=True,
                                      help="Name of dadi demographic model",)
    # model will dictate params_list, func, and logs
    generate_data_parser.add_argument('--n_samples', type=_pos_int,
                                      required=True,
                                      help="How many FS to generate",)
    generate_data_parser.add_argument('--sample_sizes', type=_pos_int,
                                      nargs='+', action='store',
                                      dest='sample_sizes',
                                      required=True,
                                      help="Sample sizes of populations",)
    generate_data_parser.add_argument('--outdir',
                                      type=str, required=True,
                                      help="Path to save generated data")
    generate_data_parser.add_argument('--grids', type=_pos_int,
                                      nargs=3, help='Sizes of grids',
                                      default=[40, 50, 60])
    generate_data_parser.add_argument('--theta', nargs='*', type=_pos_int,
                                      action='store',
                                      dest='theta',
                                      help="Factor to multiply FS with",
                                      default=[1])
    generate_data_parser.add_argument('--sampling',
                                      action='store_true',
                                      help="Whether to sample FS when \
                                          theta > 1",
                                      default=True)
    generate_data_parser.add_argument('--normalize', action='store_true',
                                      help="Whether to normalize FS when \
                                           theta > 1",
                                      default=True)
    generate_data_parser.add_argument('--bootstrap', action='store_true',
                                      help="Whether to generate bootstrap \
                                           FS data",
                                      default=False)
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
    train_parser.add_argument("model_dir")
    train_parser.add_argument("data_dir")
    # add flags for specifying different mlpr hyperparam
    # train_parser.add_argument("--epochs", type=int, default=10)

    # subcommand for predict
    predict_parser = subparsers.add_parser(
        "predict", help='Use trained MLPR to predict demographic parameters \
            from frequency spectra data')
    predict_parser.set_defaults(func=run_predict)
    # need to handle dir for multiple models for mapie
    # single dir for sklearn models
    predict_parser.add_argument("model_dir")
    # predict_parser.add_argument("output_dir")
    # predict_parser.add_argument("text_dir")
    # predict_parser.add_argument("--evaluate", dest='reference_dir')

    # subcommand for stat
    stat_parser = subparsers.add_parser(
        "stat", help='Get prediction scores and statistics')

    # subcommand for plot
    plot_parser = subparsers.add_parser(
        "plot", help='Plot prediction results and statistics')

    # subcommand for tune
    tune_parser = subparsers.add_parser(
        "tune", help='MLPR hyperparam tuning with hyperband')

    return parser


def _pos_int(input_int):
    """
    Check positive int."""
    if int(input_int) < 0:
        raise argparse.ArgumentTypeError(f"{input_int} is not a positive int")
    return int(input_int)


def main(arg_list=None):
    """Main program"""
    parser = dadi_ml_parser()
    args = parser.parse_args(arg_list)
    args.func(args)


if __name__ == "__main__":
    # this will be commented out in the final code
    # since main() will be accessed from the command line
    main()
