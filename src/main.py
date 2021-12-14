"""Command-line interface setup for dadi-ml"""
import argparse

def run_generate_data():
    from generate_data import generate_data
    import dadi_dem_models
    ...


def run_train():
    from train import train
    ...


def run_predict():
    from predict import predict
    ...


def dadi_ml_parser():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Machine learning application for dadi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(required=True)

    ## subcommand for generate_data
    generate_data_parser = subparsers.add_parser('generate_data')
    generate_data_parser.set_defaults(func=run_generate_data)
    generate_data_parser.add_argument('model',
                                      choices=[])  
    # model will dictate params_list, func, and logs
    generate_data_parser.add_argument('ns')
    generate_data_parser.add_argument('pts_l')  
    generate_data_parser.add_argument('theta')
    generate_data_parser.add_argument('n_samples')
    generate_data_parser.add_argument('--sample')
    generate_data_parser.add_argument('--normalize')
    generate_data_parser.add_argument('--bootstrap')
    generate_data_parser.add_argument('--n_bstr')
    generate_data_parser.add_argument('--n_cpu')

    
    ## subcommand for train
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=run_train)
    train_parser.add_argument("model_dir")
    train_parser.add_argument("data_dir")
    # add flags for specifying different mlpr hyperparam
    # train_parser.add_argument("--epochs", type=int, default=10)


    ## subcommand for predict
    # predict_parser = subparsers.add_parser("predict")
    # predict_parser.set_defaults(func=run_predict)
    ## need to handle dir for multiple models for mapie
    ## single dir for sklearn models
    # predict_parser.add_argument("model_dir")
    # predict_parser.add_argument("output_dir")
    # predict_parser.add_argument("text_dir")
    # predict_parser.add_argument("--evaluate", dest='reference_dir')


    ## subcommand for plot

    return parser


def main(arg_list=None):
    """Main program"""
    parser = dadi_ml_parser()
    args = parser.parse_args(arg_list)
    args.func(args)


if __name__ == "__main__":
    main()
