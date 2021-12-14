""" Tests for generate_data.py """
import os


def test_exists():
    """ Program exists """

    PRG = '../src/generate_data.py'
    assert os.path.isfile(PRG)
