""" Tests for generate_data.py """
import os
import dadi
import dadinet.dadi_dem_models as models
from dadinet.generate_data import generate_fs


def test_exists():
    """ Test program exists """

    PRG = '../dadinet/generate_data.py'
    assert os.path.isfile(PRG)


def run(model, sample_size, thetas, n_samples):
    '''Template method for testing different models, sample sizes, thetas'''

    pts = [40, 50, 60]
    dem, dem_params, p_logs = model(n_samples)
    data = generate_fs(dem, dem_params, p_logs, thetas, sample_size, pts)

    # check that the method generates one data dict per theta value
    assert len(data) == len(thetas)

    # check that the number of FS in one data dict is as specified
    assert len(data[0]) == n_samples

    # check that each dataset format is a dict
    assert isinstance(data[0], dict)

    # check that the values in the data dict is a dadi FS object
    assert isinstance(list(data[0].values())[0], dadi.Spectrum_mod.Spectrum)

    # check that the FS have the correct number of populations
    assert len(list(data[0].values())[0].shape) == len(sample_size)

    # check that the FS have the correct population sample size for each pop
    for i, pop_size in enumerate(list(data[0].values())[0].shape):
        assert pop_size == sample_size[i] + 1


def test_run_two_epoch():
    '''Generate three 10 FS datasets for the two_epoch model
    with one population sample size 20 and three thetas: 1, 1e4, 1e2'''

    run(models.two_epoch, [20], [1, 10000, 100], 10)


def test_run_growth():
    '''Generate two 20 FS datasets for the growth model
    with one population sample size 100 and two thetas: 1 and 1e3'''

    run(models.growth, [100], [1, 1000], 20)


def test_run_split_mig():
    '''Test generate one 5 FS datasets for the split migration model
    with two populations sample sizes 10, 15 and theta 100'''

    run(models.split_mig, [10, 15], [100], 5)


def test_run_IM():
    '''Test generate one 3 FS datasets for the IM model
    with two populations sample sizes 30, 21 and theta 1000'''

    run(models.IM, [30, 21], [1000], 3)


# test generate data for bootstrap data
