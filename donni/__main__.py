"""Command-line interface setup for donni"""
import argparse
import pickle
import sys
import os
import dadi
import numpy as np
from scipy.stats._distn_infrastructure import rv_frozen as distribution
from donni.dadi_dem_models import get_model, get_param_values
from donni.generate_data import generate_fs, get_hyperparam_tune_dict, fs_quality_check
from donni.train import prep_data, tune, train, default_hyperparams
from donni.infer import infer, prep_fs_for_ml, irods_download, irods_cleanup, project_fs
from donni.validate import validate


# run_ methods for importing methods from other modules
def run_generate_data(args):
    """Method to generate data given inputs from the
    generate_data subcommand"""

    if args.save_individual_fs:
        # check if outdir is provided
        if args.outdir is None:
            sys.exit(
                "donni generate_data: error: "
                "the following arguments are required:"
                " --outdir when using --save_individual_fs"
            )

    # get dem function and params specifications for model
    dadi_func, param_names, logs = get_model(args.model, args.model_file, args.folded)
    # get demographic param values
    params_list = get_param_values(param_names, args.n_samples, args.seed)

    if not args.generate_tune_hyperparam_only:
        # generate data
        data, qual = generate_fs(
            dadi_func,
            params_list,
            logs,
            args.theta,
            args.sample_sizes,
            args.grids,
            args.non_normalize,
            args.no_sampling,
            args.folded,
            args.bootstrap,
            args.n_bstr,
            args.n_cpu,
        )

        # output fs quality check results
        if not args.no_fs_qual_check:
            fs_quality_check(qual, args.outfile, params_list, param_names, logs)

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
            pickle.dump(true_log_params, open(f"{args.outdir}/true_log_params", "wb"))

        # save data dict as one pickled file (default)
        pickle.dump(data, open(args.outfile, "wb"))

    # generate and save hyperparams dict for tuning
    if args.generate_tune_hyperparam_only or args.generate_tune_hyperparam:
        tune_dict = get_hyperparam_tune_dict(args.sample_sizes)
        pickle.dump(tune_dict, open(f"{args.hyperparam_outfile}", "wb"))

        # output text file for details of hyper param tune dict
        with open(f"{args.hyperparam_outfile}.txt", "w") as fh:
            for hyperparam, option in tune_dict.items():
                if isinstance(option, distribution):
                    distribution_name = vars(option.dist)["name"]
                    distribution_vals = vars(option)["args"]
                    fh.write(
                        f"{hyperparam}: {distribution_name}, " f"{distribution_vals}\n"
                    )
                else:
                    fh.write(f"{hyperparam}: {option}\n")


def run_train(args):
    """Method to train MLPR given inputs from the train subcommand"""

    # Load training data
    data = pickle.load(open(args.data_file, "rb"))
    # parse data into input and corresponding labels
    X_input, y_label = prep_data(data, mapie=args.multioutput)
    # make dir to save trained MLPs
    try:
        os.makedirs(args.mlpr_dir)
    except FileExistsError:
        pass
    # iterate through each param in the demographic model
    for index, param in enumerate(y_label):
        # define path to save trained model
        model_path = f"{args.mlpr_dir}/param_{index}_predictor.keras"
        # get tuned hyperparams
        if args.tune or args.tune_only:
            best_hp = tune(X_input, np.array(param))
        # or generate a default
        else:
            best_hp = default_hyperparams()
        train(best_hp, X_input, np.array(param), model_path)


def _load_trained_mlpr(args):
    """Helper method to read in trained MLPR models for predict and validate"""

    mlpr_list = []
    mapie = True
    filename_list = sorted(os.listdir(args.mlpr_dir))
    if "param_all_predictor" in filename_list:
        mapie = False  # this is the sklearn case
        mlpr = pickle.load(
            open(os.path.join(args.mlpr_dir, "param_all_predictor"), "rb")
        )
        mlpr_list = [mlpr]
    else:
        for filename in filename_list:
            if filename.startswith("param") and filename.endswith("predictor"):
                mlpr = pickle.load(open(os.path.join(args.mlpr_dir, filename), "rb"))
                mlpr_list.append(mlpr)

    # need to get logs to de-log prediction
    _, param_names, logs = get_model(
        args.model, args.model_file, args.folded
    )  # now included misid

    return mlpr_list, mapie, logs, param_names


def run_infer(args):
    """Method to get prediction given inputs from the
    predict subcommand"""
    # open input FS from file
    fs = dadi.Spectrum.from_file(args.input_fs)
    args.folded = fs.folded
    if args.cleanup:
        irods_cleanup(args.model, fs.sample_sizes, args.folded)
        return
    if args.mlpr_dir != None:
        # load trained MLPRs and demographic model logs; TODO: remove for cloud support
        mlpr_list, mapie, logs, param_names = _load_trained_mlpr(args)
        qc_dir = False
    else:
        fs = project_fs(fs)
        ss = fs.sample_sizes
        args.mlpr_dir, qc_dir = irods_download(
            args.model, ss, args.folded, args.download_dir
        )
        # load trained MLPRs and demographic model logs; TODO: remove for cloud support
        mlpr_list, mapie, logs, param_names = _load_trained_mlpr(args)
    # load func
    func, _, _ = get_model(args.model, args.model_file, args.folded)
    cis_list = sorted(args.cis)
    # infer params using input FS
    pred, theta, cis = infer(mlpr_list, func, fs, logs, mapie=mapie, cis=cis_list)
    # write output
    if args.output_prefix:
        output_stream = open(args.output_prefix, "w")
    else:
        output_stream = sys.stdout
    pred.append(theta)
    ci_names = []
    for i, ci in enumerate(cis_list):
        for j, param in enumerate(param_names):
            ci_names.append(param + "_lb_" + str(ci))
            ci_names.append(param + "_ub_" + str(ci))
            pred.append(cis[j][i][0])
            pred.append(cis[j][i][1])
    print_names = param_names + ["theta"] + ci_names
    print("\n***Inferred demographic model parameters***")
    # print parameter names
    print("# ", end="", file=output_stream)
    print(*print_names, sep="\t", file=output_stream)
    # print prediction
    print(*pred, sep="\t", file=output_stream)
    print(file=output_stream)  # newline
    # print readable intervals
    print(f"{'# CIs: ':<10}", end="", file=output_stream)
    for ci in cis_list:
        print(f"|----------{ci}----------|", end="\t", file=output_stream)
    print(file=output_stream)
    for i, param in enumerate(param_names):
        print(f"{'# ' + param + ': ':<10}", end="", file=output_stream)
        for ci in cis[i]:
            print(f"[{ci[0]:10.6f}, {ci[1]:10.6f}]", end="\t", file=output_stream)
        print(file=output_stream)
    if args.output_prefix:
        output_stream.close()
    if qc_dir is not False:
        print(
            f"\nCheck the plots in {qc_dir} for accuracy scores of the downloaded model."
        )
    if theta is np.nan:
        print(
            f"\nWARNING: Theta is not defined. Check inferred demographic model parameters for negative values."
        )
    if args.export_dadi_cli is not None:
        pts_l = pts_l_func(fs.sample_sizes)
        fid = open(args.export_dadi_cli + ".donni.pseudofit", "w")
        fid.write("# {0}\n".format(" ".join(sys.argv)))
        fid.write(f"# grid points used: {pts_l}\n")
        fid.write("# Log(likelihood)\t{0}\ttheta\n".format("\t".join(param_names)))
        fid.write(
            "-0\t{0}\t".format(
                "\t".join([str(ele) for ele in pred[: len(param_names) + 1]])
            )
        )


def run_validate(args):
    # load test fs set
    test_dict = pickle.load(open(args.test_dict, "rb"))
    # prepare fs in test_dict for ml prediction:
    # check that fs is normalized and masked entries set to 0
    prep_test_dict = {}
    for params_key in test_dict:
        prep_test_dict[params_key] = prep_fs_for_ml(test_dict[params_key])

    # parse test dict into test FS and corresponding labels
    X_test, y_test = prep_data(prep_test_dict, mapie=True)

    # load mlpr dir name list
    filename_list = sorted(os.listdir(args.mlpr_dir))

    # make result dir to save results
    try:
        os.makedirs(args.results_dir)
    except FileExistsError:
        pass

    # get logs to de-log prediction
    _, param_names, logs = get_model(
        args.model, args.model_file, args.folded
    )  # now included misid

    # result file
    plot_prefix = os.path.join(args.results_dir, args.plot_prefix)

    validate(
        filename_list, args.mlpr_dir, X_test, y_test, param_names, logs, plot_prefix
    )


# helper methods for custom type checks and parsing
def _pos_int(input_int):
    """
    Check positive integer
    """

    if int(input_int) < 0:
        raise argparse.ArgumentTypeError(f"{input_int} is not a positive int")
    return int(input_int)


def donni_parser():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description="Demography Optimization via Neural Network Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    # subcommand for generate_data
    generate_data_parser = subparsers.add_parser(
        "generate_data",
        help="Simulate allele frequency data from demographic history models",
    )
    generate_data_parser.set_defaults(func=run_generate_data)

    generate_data_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of dadi demographic model",
    )
    generate_data_parser.add_argument(
        "--model_file",
        type=str,
        help="Name of file containing custom dadi\
                                         demographic model(s)",
    )
    # --model will dictate params_list, func, logs, and param_names
    generate_data_parser.add_argument(
        "--n_samples",
        type=_pos_int,
        required=True,
        help="How many FS to generate",
    )
    generate_data_parser.add_argument(
        "--sample_sizes",
        type=_pos_int,
        nargs="+",
        required=True,
        help="Sample sizes of populations",
    )
    generate_data_parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="Path to save generated data and\
                                         associated quality check file",
    )
    generate_data_parser.add_argument(
        "--save_individual_fs",
        action="store_true",
        help="Save individual FS as a file\
                                        instead of together in one dictionary",
    )
    generate_data_parser.add_argument(
        "--outdir", type=str, help="Dir to save individual FS"
    )
    generate_data_parser.add_argument(
        "--grids", type=_pos_int, nargs=3, help="Sizes of grids", default=None
    )
    generate_data_parser.add_argument(
        "--theta", type=_pos_int, help="Factor to multiply FS with", default=1
    )
    generate_data_parser.add_argument(
        "--seed", type=_pos_int, help="Seed for reproducibility"
    )
    generate_data_parser.add_argument(
        "--non_normalize", action="store_false", help="Don't normalize FS"
    )
    generate_data_parser.add_argument(
        "--no_sampling", action="store_false", help="Don't sample FS when theta > 1"
    )
    generate_data_parser.add_argument(
        "--folded", action="store_true", help="Whether to fold FS"
    )
    generate_data_parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Whether to generate bootstrap\
                                           FS data",
    )
    generate_data_parser.add_argument(
        "--n_bstr",
        type=_pos_int,
        help="Number of bootstrap FS to generate\
                                         for each FS (if bootstrap)",
        default=200,
    )
    generate_data_parser.add_argument(
        "--n_cpu", type=_pos_int, help="Number of CPUs to use"
    )
    generate_data_parser.add_argument(
        "--no_fs_qual_check",
        action="store_true",
        help="Turn off default FS quality check",
    )
    generate_data_parser.add_argument(
        "--generate_tune_hyperparam",
        action="store_true",
        help="Generate hyperparam spec for\
                                         tuning",
    )
    generate_data_parser.add_argument(
        "--generate_tune_hyperparam_only",
        action="store_true",
        help="Generate hyperparam spec for tuning\
                                      only without generating any data",
    )
    generate_data_parser.add_argument(
        "--hyperparam_outfile",
        type=str,
        default="param_dict_tune",
        help="Path to save hyperparam dict\
                                         for tuning",
    )

    # subcommand for train
    train_parser = subparsers.add_parser(
        "train", help="Train MLPR with simulated allele frequency data"
    )
    train_parser.set_defaults(func=run_train)
    train_parser.add_argument(
        "--data_file", type=str, required=True, help="Path to input training data"
    )
    train_parser.add_argument(
        "--mlpr_dir",
        type=str,
        required=True,
        help="Path to save output trained MLPR(s)",
    )
    train_parser.add_argument(
        "--multioutput",
        action="store_false",
        help="Train multioutput sklearn MLPR instead of\
                                  default mapie single-output MLPRs",
    )
    train_parser.add_argument(
        "--tune",
        action="store_true",
        help="Whether to try a range of hyperparameters\
                                   to find the best performing MLPRs",
    )
    train_parser.add_argument(
        "--tune_only",
        action="store_true",
        help="When use will not refit to train\
                                 on the full data set after tuning",
    )
    # subcommand for infer
    infer_parser = subparsers.add_parser(
        "infer",
        help="Infer demographic history parameters\
                        from allele frequency with trained MLPRs",
    )
    infer_parser.set_defaults(func=run_infer)
    # need to handle dir for multiple models for mapie
    # single dir for sklearn models
    infer_parser.add_argument(
        "--input_fs",
        type=str,
        required=True,
        help="Path to FS file for generating\
                                     inference",
    )
    infer_parser.add_argument(
        "--model", type=str, required=True, help="Name of dadi demographic model"
    )
    infer_parser.add_argument(
        "--download_dir",
        type=str,
        required=False,
        help="Path to saved, trained MLPR(s) downloaded from the University of Arizona CyVerse Data Store.",
    )
    infer_parser.add_argument(
        "--mlpr_dir",
        type=str,
        required=False,
        help="Path to saved, trained MLPR(s). Required if user is not downloading MLPRs for inference.",
    )
    # optional
    infer_parser.add_argument(
        "--cis",
        type=_pos_int,
        nargs="+",
        default=[95],
        help="Optional list of values for\
                                    confidence intervals,\
                                    e.g., [80 90 95]; default [95]",
    )
    infer_parser.add_argument(
        "--model_file",
        type=str,
        help="Name of file containing custom dadi\
                                     demographic model(s)",
    )
    infer_parser.add_argument(
        "--output_prefix",
        type=str,
        help="Optional output file to write out results\
                                   (default stdout)",
    )
    infer_parser.add_argument(
        "--export_dadi_cli",
        type=str,
        default=None,
        help='Optional. Pass a file name to generate a\
                                dadi-cli bestfit file to analyze with dadi-cli.\
                                Filename will end in ".donni.pseudofit".',
    )
    infer_parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Optional. Delete the default directory for a given model configuration's\
                                MLPRs and QC files downloaded from Cyverse.",
    )

    # subcommand for validate
    validate_parser = subparsers.add_parser(
        "validate", help="Validate trained MLPRs inference accuracy and CI coverage"
    )
    validate_parser.set_defaults(func=run_validate)

    validate_parser.add_argument(
        "--mlpr_dir", type=str, required=True, help="Path to trained MLPR(s)"
    )
    validate_parser.add_argument(
        "--test_dict", type=str, required=True, help="Path to test data dictionary file"
    )
    validate_parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory to save output plots"
    )
    validate_parser.add_argument(
        "--plot_prefix", type=str, required=True, help="Prefix for plot filenames"
    )
    validate_parser.add_argument(
        "--model", type=str, required=True, help="Name of dadi demographic model"
    )
    # optional
    validate_parser.add_argument(
        "--model_file",
        type=str,
        help="Name of file containing custom dadi\
                                 demographic model(s)",
    )
    validate_parser.add_argument('--folded', action="store_true",
                                help="Specify if the test FS is folded")

    return parser


def main(arg_list=None):
    """Main program"""

    parser = donni_parser()
    args = parser.parse_args(arg_list)
    args.func(args)
