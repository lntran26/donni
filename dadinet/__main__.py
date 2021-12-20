"""Command-line interface setup for dadi-ml"""
import argparse


def run_generate_data(args):
    from generate_data import generate_data
    import dadi_dem_models

    # method to get function from model name
    args.model
    ...


def run_train(args):
    from train import train
    ...


def run_predict(args):
    from predict import predict
    ...


dadi_dem_models = ['two_epoch', 'growth', 'split_mig', 'IM']


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
    generate_data_parser.add_argument('--model', required=True, type=str,
                                      choices=dadi_dem_models)
    # model will dictate params_list, func, and logs
    generate_data_parser.add_argument('--ns', required=True)
    generate_data_parser.add_argument('--grids', required=True)
    generate_data_parser.add_argument('--theta', nargs='*', type=_pos_int,
                                      action='store',
                                      dest='theta', default=[1])
    generate_data_parser.add_argument('n_samples')
    generate_data_parser.add_argument('--sample')
    generate_data_parser.add_argument('--normalize')
    generate_data_parser.add_argument('--bootstrap')
    generate_data_parser.add_argument('--n_bstr')
    generate_data_parser.add_argument('--n_cpu')
    generate_data_parser.add_argument('outdir',
                                      type=str,
                                      help='Path to save generated data')

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
        raise argparse.ArgumentTypeError("Theta must be positive")
    return int(input_int)


def main(arg_list=None):
    """Main program"""
    parser = dadi_ml_parser()
    args = parser.parse_args(arg_list)
    args.func(args)


if __name__ == "__main__":
    main()
