""" Tests for generate_data.py """
import os
import random
import numpy as np
import pytest
import dadi
import dadinet.dadi_dem_models as models
from dadinet.generate_data import generate_fs


def test_exists():
    """ Test program exists """

    PRG = '../dadinet/generate_data.py'
    assert os.path.isfile(PRG)


def run(model, sample_size, theta, n_samples, norm=True, sampling=True):
    '''Template method for testing different models, sample sizes, thetas'''

    grids = [40, 50, 60]
    dem, dem_params, p_logs = model(n_samples)
    data = generate_fs(dem, dem_params, p_logs, theta,
                       sample_size, grids, norm, sampling)

    # check that output dataset format is a dict
    assert isinstance(data, dict)

    # check that the number of FS in output dataset dict is as specified
    assert len(data) == n_samples

    all_fs = list(data.values())
    # check that the values in the data dict is a dadi FS object
    assert all(isinstance(fs, dadi.Spectrum_mod.Spectrum) for fs in all_fs)

    # check that the FS have the correct number of populations
    assert all(len(fs.shape) == len(sample_size) for fs in all_fs)

    # check that all FS have the correct population sample size for each pop
    for fs in all_fs:
        for i, pop_size in enumerate(fs.shape):
            assert pop_size == sample_size[i] + 1

    all_sum = [fs.sum() for fs in all_fs]
    if norm:  # check that all FS are normalized
        all_sum_expected = np.full(len(all_sum), 1.)
        np.testing.assert_allclose(all_sum, all_sum_expected)

    elif not sampling:  # non-normalized and not sampling
        # get a random fs params set in the list
        test_p = random.choice(dem_params)
        test_p_logs = [10**test_p[i] if p_logs[i] else test_p[i]
                       for i in range(len(p_logs))]
        # make expected fs with dadi
        func_ex = dadi.Numerics.make_extrap_func(dem)
        expected_fs = theta*func_ex(test_p_logs, sample_size, grids)
        # check that expected fs is the same
        np.testing.assert_array_equal(data[test_p], expected_fs)


def test_run_two_epoch():
    '''Generate 10 FS datasets for the two_epoch model
    with one population sample size 160 and theta 1'''

    run(models.two_epoch, [160], 1000, 10)


def test_run_two_epoch_non_norm():
    '''Generate non-normalized FS for the two_epoch model'''

    run(models.two_epoch, [20], 10000, 5, sampling=False, norm=False)


def test_run_growth():
    '''Generate 20 FS datasets for the growth model
    with one population sample size 80 and theta 1000'''

    run(models.growth, [80], 1000, 20)


def test_run_split_mig():
    '''Generate 5 FS datasets for the split migration model
    with two populations sample sizes 40, 40 and theta 100'''

    run(models.split_mig, [40, 40], 100, 5)


def test_run_split_mig_non_norm():
    '''Generate non-normalized FS for the split_mig model'''

    run(models.split_mig, [10, 10], 1000, 2, sampling=False, norm=False)


def test_run_IM():
    '''Generate 3 FS datasets for the IM model
    with two populations sample sizes 20, 20 and theta 10000'''

    run(models.IM, [20, 20], 10000, 3)


def run_bootstrap(model, sample_size, theta, n_samples, n_bstr):
    '''Template method for testing generating bootstrap data'''

    grids = [40, 50, 60]
    dem, dem_params, p_logs = model(n_samples)
    data = generate_fs(dem, dem_params, p_logs, theta, sample_size,
                       grids, bootstrap=True, n_bstr=n_bstr)

    # check that output dataset format is a dict
    assert isinstance(data, dict)

    # check that the number of FS in output dataset dict is as specified
    assert len(data) == n_samples

    all_vals = list(data.values())
    # check that the values in the data dict is a list of length 2
    assert all(isinstance(val, list) and len(val) == 2 for val in all_vals)

    all_fs, all_bstr_fs = zip(*all_vals)
    # check that the first value is a fs
    assert all(isinstance(fs, dadi.Spectrum_mod.Spectrum) for fs in all_fs)

    for bstr in all_bstr_fs:
        # check that the second value is a list of the correct length
        assert isinstance(bstr, list) and len(bstr) == n_bstr

        # check that this list contains fs objects
        assert all(isinstance(fs, dadi.Spectrum_mod.Spectrum) for fs in bstr)


def test_run_bstr_theta_1():
    '''Test raising SystemExit exception when trying to
    generate bootstrap data with theta = 1'''
    grids = [40, 50, 60]
    dem, dem_params, p_logs = models.two_epoch(5)
    with pytest.raises(SystemExit):
        generate_fs(dem, dem_params, p_logs, 1, [
                    20], grids, bootstrap=True, n_bstr=10)


def test_run_two_epoch_bstr():
    '''Generate 10 bootstrap datasets (theta=1000) for 5 FS datasets
    of the two_epoch model with one population sample size 20'''

    run_bootstrap(models.two_epoch, [20], 1000, 5, 10)


def test_run_split_mig_bstr():
    '''Generate 5 bootstrap datasets (theta=100) for 3 FS datasets
    of the split_mig model with two population sample sizes 20, 20'''

    run_bootstrap(models.split_mig, [20, 20], 100, 3, 5)
