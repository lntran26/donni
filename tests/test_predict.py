""" Tests for predict.py """
import os


def test_exists():
    """ Test program exists """

    PRG = '../dadinet/predict.py'
    assert os.path.isfile(PRG)
