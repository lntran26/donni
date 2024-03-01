""" Tests for train.py """
import os
import pickle
from donni.train import *


def test_exists():
    """ Test program exists """

    PRG = 'donni/train.py'
    assert os.path.isfile(PRG)


def test_prep_data():
    """ Test prep_data() method """

    for data_file in ['two_epoch_500', 'split_mig_100_subset']:
        data = pickle.load(open(f'tests/test_data/{data_file}', 'rb'))
        X, y = prep_data(data)
        # test that X has the correct n_samples
        assert len(X) == len(data)
        # test that y contains the same number of inner list per dem param
        assert any(len(key) == len(y) for key in list(data.keys()))
        # test that each inner list in y has the correct n_samples
        assert all(len(y_unpack) == len(X) for y_unpack in y)

        # test using sklearn instead of mapie
        X, y = prep_data(data, mapie=False)
        # test that X has the correct n_samples
        assert len(X) == len(data)
        # test that y contains only 1 inner list
        assert len(y) == 1
        # test that the inner list in y has the correct n_samples
        assert len(y[0]) == len(X)
        # test that each member of the inner list contains the
        # correct number of dem params
        for val, key in zip(y[0], list(data.keys())):
            assert len(val) == len(key)
