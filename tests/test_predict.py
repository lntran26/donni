import dadi
import pickle
import pytest
import os
import numpy as np
from dadinet.predict import project_fs, predict


@pytest.fixture
def models_list():
    mlpr_dir = "test_models/split_mig_tuned_20_20"
    mlpr_list = []
    for filename in sorted(os.listdir(mlpr_dir)):
        mlpr = pickle.load(open(os.path.join(mlpr_dir, filename), 'rb'))
        mlpr_list.append(mlpr)
    return mlpr_list


@pytest.fixture
def split_mig_fs():
    p = (0.5, 0.2, 3, 2) # nu1, nu2, T, m
    func = getattr(dadi.Demographics2D, "split_mig")
    pts_l = [40, 50]
    input_size = (40, 30)
    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p, input_size, pts_l)
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
    func = getattr(dadi.Demographics1D, "two_epoch")
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
    func = getattr(dadi.Demographics2D, "split_mig")
    pts_l = [40, 50]

    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p, input_size, pts_l)

    projected_fs = project_fs(fs)
    np.testing.assert_array_equal(projected_fs, fs.project(exp_size))


@pytest.mark.parametrize("pis",
                        [[95],
                         [95, 80, 70]])
def test_predict_split_mig(models_list, split_mig_fs, pis):
    func = getattr(dadi.Demographics2D, "split_mig")
    logs = [True, True, False, False, False]
    pred_list, theta, pi_list = predict(models_list, func, split_mig_fs, logs, pis=pis)
    pred_list = np.array(pred_list)
    pi_list = np.array(pi_list)

    assert pred_list.shape == (5,)
    assert pi_list.shape == (5, len(pis), 2)