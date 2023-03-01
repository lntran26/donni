import dadi
import pickle
import pytest
import os
import glob
import numpy as np
from donni.infer import project_fs, infer


@pytest.fixture
def models_list():
    return [pickle.load(open(filename,'rb')) for filename in glob.glob("test_models/split_mig_tuned_20_20/param_*_predictor")]


@pytest.fixture
def split_mig_fs():
    p0 = [1, 1, 0.1, 0.5]
    ns = [40, 30]
    pts_l = [50 ,60 ,70]
    func = dadi.Demographics2D.split_mig
    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p0, ns, pts_l)
    return fs


@pytest.mark.parametrize("input_size, exp_size",
                        [((93,), (81,)),
                         ((81, 81), (81, 81)),
                         ((40, 15), (11, 11)),
                         ((23, 18, 40), (11, 11, 11))])
def test_project_fs_fake(input_size, exp_size):
    fs_array = np.zeros(input_size)
    fs = dadi.Spectrum(fs_array)
    projected_fs = project_fs(fs)
    assert projected_fs.shape == exp_size


@pytest.mark.parametrize("input_size, exp_size",
                        [((19,), (10,)),
                         ((20,), (20,)),
                         ((92,), (80,))])
def test_project_fs_1d(input_size, exp_size):
    p = (1, 4) # nu, T
    func = dadi.Demographics1D.two_epoch
    pts_l = [40]

    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p, input_size, pts_l)

    projected_fs = project_fs(fs)
    np.testing.assert_array_equal(projected_fs, fs.project(exp_size))


@pytest.mark.parametrize("input_size, exp_size",
                        [((25, 54), (20, 20)),
                         ((40, 40), (40, 40))])
def test_project_fs_2d(input_size, exp_size):
    p = (1, 1, 4, 5) # nu1, nu2, T, m
    func = dadi.Demographics2D.split_mig
    pts_l = [40, 50]

    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p, input_size, pts_l)

    projected_fs = project_fs(fs)
    np.testing.assert_array_equal(projected_fs, fs.project(exp_size))

@pytest.mark.skip("Test not working")
@pytest.mark.parametrize("pis",
                        [[95],
                         [95, 80, 70]])
def test_infer_split_mig(models_list, split_mig_fs, pis):
    func = dadi.Demographics2D.split_mig
    logs = [True, True, False, False, False]
    pred_list, theta, pi_list = infer(models_list, func, split_mig_fs, logs, pis=pis)
    pred_list = np.array(pred_list)
    pi_list = np.array(pi_list)

    assert pred_list.shape == (5,)
    assert pi_list.shape == (5, len(pis), 2)
