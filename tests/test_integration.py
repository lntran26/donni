'''
Intergration test for the entire program
'''
import os
import re
import random
import string
from subprocess import getstatusoutput, getoutput
import pickle
import shutil

PRG = 'donni'


def random_string():
    """generate a random filename"""

    return 'tests/'+''.join(random.choices(string.ascii_lowercase + string.digits, k=5))


def test_usage():
    """ Prints program usage """

    for flag in ['', '-h', '--help']:
        out = getoutput(f'{PRG} {flag}')
        assert re.match("usage", out, re.IGNORECASE)


def test_usage_subcommand():
    """ Prints subcommand usage """

    for subcommand in ['generate_data', 'train', 'infer', 'plot']:
        for flag in ['', '-h', '--help']:
            out = getoutput(f'{PRG} {subcommand} {flag}')
            assert re.match("usage", out, re.IGNORECASE)


# test generate_data subcommand
def run_generate_data_sub(args, args_expected):
    """Template method for testing generate_data subcommand"""

    outfile = random_string()
    try:
        rv, _ = getstatusoutput(
            f'{PRG} generate_data {" ".join(args)} --outfile {outfile}')

        # check that program executed without errors
        assert rv == 0
        # check that program does not produce any output to CLI
        # Temporarily skipping what output is figured out
        # assert out.strip() == ""
        # check that program produces an output file
        assert os.path.isfile(outfile)
        # check that output file has the correct format
        # this is only done briefly here as test_generate_data.py
        # is already checking output data more rigorously
        data = pickle.load(open(outfile, 'rb'))
        assert len(data) == args_expected['n_samples']

    finally:  # remove output files
        if os.path.isfile(outfile):
            os.remove(outfile)
        if os.path.isfile(f'{outfile}_quality.txt'):
            os.remove(f'{outfile}_quality.txt')


def test_run_generate_data_sub_1():
    '''First generate_data test'''

    args = ['--model two_epoch', '--n_samples 5',
            '--sample_sizes 10', '--theta 1000']
    args_expected = {'n_samples': 5,
                     'sample_sizes': [10], 'thetas': 1000}
    run_generate_data_sub(args, args_expected)


def test_run_generate_data_sub_2():
    '''Second generate_data test'''

    args = ['--model split_mig', '--n_samples 1',
            '--sample_sizes 15 20']
    args_expected = {'n_samples': 1,
                     'sample_sizes': [15, 20], 'thetas': 1}
    run_generate_data_sub(args, args_expected)


def test_run_generate_data_sub_folded():
    '''Third generate_data test for folded flag'''

    args = ['--model split_mig', '--n_samples 10',
            '--sample_sizes 20 20', '--folded']
    args_expected = {'n_samples': 10, 'sample_sizes': [20, 20],
                     'thetas': 1, 'folded': True}
    run_generate_data_sub(args, args_expected)


def test_run_generate_data_bstr():
    '''Generate bootstrap data test'''

    args = ['--model two_epoch', '--n_samples 5',
            '--sample_sizes 10', '--theta 100', '--bootstrap']
    args_expected = {'n_samples': 5,
                     'sample_sizes': [10], 'thetas': 1000}
    run_generate_data_sub(args, args_expected)


# test train subcommand
def run_train_sub(args):
    """Template method for testing train subcommand"""
    outdir = random_string()
    os.mkdir(outdir)
    try:
        rv, _ = getstatusoutput(
            f'{PRG} train {" ".join(args)} --mlpr_dir {outdir}')

        # check that program executed without errors
        assert rv == 0
        # check that program produces some output file(s)
        assert len(os.listdir(outdir)) != 0
        # can expand this later to test correct number of outfiles
        # for each use case

    finally:  # remove output dir
        shutil.rmtree(outdir, ignore_errors=True)


def test_run_train_sub_1():
    '''First train test: provide data and run default:
    Generate mapie MLPRs without tuning, use default hyperparams'''

    args = ['--data_file tests/test_data/two_epoch_500']
    run_train_sub(args)


# test infer subcommand
def run_infer_sub(args):
    """Template method for testing train subcommand"""
    outdir = random_string()
    os.mkdir(outdir)
    try:
        rv, _ = getstatusoutput(
            f'{PRG} infer {" ".join(args)} --output_prefix {outdir}/test_infer_output --export_dadi_cli {outdir}/test_infer_output')

        # check that program executed without errors
        assert rv == 0
        # check that program produces some output file(s)
        assert len(os.listdir(outdir)) != 0
        # can expand this later to test correct number of outfiles
        # for each use case

    finally:  # remove output dir
        shutil.rmtree(outdir, ignore_errors=True)


def test_run_infer():
    '''
    '''
    args = ['--input_fs tests/test_data/split_mig_syn.fs', '--model split_mig']
    run_infer_sub(args)
