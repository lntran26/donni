'''
Intergration test for the entire program
'''
import re
from subprocess import getstatusoutput, getoutput


# test program usage
def test_usage():
    """ Prints program usage """

    for flag in ['', '-h', '--help']:
        out = getoutput(f'dadi-ml {flag}')
        assert re.match("usage", out, re.IGNORECASE)


# test subcommand usage
def test_usage_subcommand():
    """ Prints subcommand usage """

    for subcommand in ['generate_data', 'train', 'predict', 'plot', 'tune']:
        for flag in ['', '-h', '--help']:
            out = getoutput(f'dadi-ml {subcommand} {flag}')
            assert re.match("usage", out, re.IGNORECASE)


# test generate_data subcommand
def run_generate_data_sub(args):
    """Template method for running generate_data subcommand"""
    rv, out = getstatusoutput(f'dadi-ml generate_data {" ".join(args)}')

    # check
    assert ...


def test_run_generate_data_sub_1():
    ''''''
    ...
