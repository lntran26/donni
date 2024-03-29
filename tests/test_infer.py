import dadi
import pickle
import pytest
import os
import glob
import numpy as np
from donni.infer import *


@pytest.fixture
def split_mig_fs():
    p0 = [1, 1, 0.1, 0.5, 0.05]
    ns = [40, 30]
    pts_l = [50 ,60 ,70]
    func = dadi.Demographics2D.split_mig
    func_misid = dadi.Numerics.make_anc_state_misid_func(func)
    func_ex = dadi.Numerics.make_extrap_func(func_misid)
    fs = func_ex(p0, ns, pts_l)
    print(fs)
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


@pytest.mark.parametrize("ci_list",
                        [[95],
                         [95, 80, 70]])
def test_infer_split_mig(split_mig_fs, ci_list):
    models_dir = 'tests/test_models/split_mig_tuned_20_20'
    models_list = sorted(os.listdir(models_dir))
    split_mig_fs = project_fs(split_mig_fs)
    func = dadi.Demographics2D.split_mig
    logs = [True, True, False, False, False]
    print(models_list, func, split_mig_fs, logs)
    pred_list, theta, cis = infer(models_list, models_dir, func, split_mig_fs, logs, cis=ci_list)
    pred_list = np.array(pred_list)
    cis = np.array(cis)

    assert pred_list.shape == (5,)
    assert cis.shape == (5, len(ci_list), 2)


@pytest.mark.skip("Skip incase server is down")
def test_irods_download(capfd):
    import shutil
    try:
        shutil.rmtree("tests/temp")
    except FileNotFoundError:
        pass
    dem_model = "two_epoch"
    sample_sizes = [10]
    fold = True
    datadir = "tests/temp"
    irods_download(dem_model, sample_sizes, fold, datadir)
    assert os.path.isfile("tests/temp/two_epoch_folded_ns_10/param_01_predictor")
    assert os.path.isfile("tests/temp/two_epoch_folded_ns_10/param_02_predictor")
    assert os.path.isfile("tests/temp/two_epoch_folded_ns_10_QC/theta_1000_coverage.png")
    assert os.path.isfile("tests/temp/two_epoch_folded_ns_10_QC/theta_1000_param_01_accuracy.png")
    assert os.path.isfile("tests/temp/two_epoch_folded_ns_10_QC/theta_1000_param_02_accuracy.png")

    irods_download(dem_model, sample_sizes, fold, datadir)
    out, err = capfd.readouterr()
    assert '\n'.join([out.split('\n')[-5],out.split('\n')[-4]]) == f"Files for the requested model and configuration have already been downloaded to the temp/two_epoch_folded_ns_10 folder.\nTo redownload, delete the existing directory."

    shutil.rmtree("tests/temp")
