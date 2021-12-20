""" Tests for generate_data.py """
import os
import random
import dadi
import dadinet.dadi_dem_models as models
from dadinet.generate_data import generate_fs
# sys.path.append(os.path.join(os.getcwd(), '..'))
# import src.generate_data


def test_exists():
    """ Program exists """

    PRG = '../dadinet/generate_data.py'
    assert os.path.isfile(PRG)



def run(model, sample_size, thetas, n_samples):
    '''dem = models.function_name'''
    pts = [40, 50, 60]
    dem, dem_params, p_logs = model(n_samples)
    data = generate_fs(dem, dem_params, p_logs, thetas, sample_size, pts)
    assert len(data) == len(thetas)
    assert len(data[0]) == n_samples
    assert type(data[0]) is dict
    assert type(list(data[0].values())[0]) is dadi.Spectrum_mod.Spectrum
    assert len(list(data[0].values())[0].shape) == len(sample_size)
    if len(sample_size) == 1:
        assert list(data[0].values())[0].shape[0] == sample_size[0] + 1


# test generate data for different demographic models, multiple thetas
def test_run_two_epoch():
    '''Test generate data for two_epoch model'''
    model = models.two_epoch
    sample_size = [20]
    thetas = [1]
    n_samples = 5
    run(model, sample_size, thetas, n_samples)


# test generate data for bootstrap data
