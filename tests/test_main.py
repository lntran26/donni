"""
This might be redundant: to be decided whether to keep
Unit tests for methods in __main__.py"""

import os


def test_exists():
    """ Program exists """

    PRG = '../dadinet/__main__.py'
    assert os.path.isfile(PRG)


def test_run_generate_data():
    '''Unit test for run_generate_data'''
    ...
