import dadi
import pytest
import numpy as np
from dadinet.predict import project_fs

@pytest.mark.parametrize("input_size, exp_size",
                        [((93,), (80,)),
                         ((80, 80), (80, 80)),
                         ((40, 15), (10, 10)),
                         ((23, 18, 40), (10, 10, 10))])
def test_project_fs_fake(input_size, exp_size):
    fs_array = np.zeros(input_size)
    fs = dadi.Spectrum(fs_array)
    projected_fs = project_fs(fs)
    assert projected_fs.shape == exp_size


def test_project_fs_real():
    # 1D
    p = (1, 4) # nu, T
    input_size, expected_size = (34,), (20,)
    func = getattr(dadi.Demographics1D, "two_epoch")
    pts_l = [40]

    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p, input_size, pts_l)

    projected_fs = project_fs(fs)
    np.testing.assert_array_equal(projected_fs, fs.project((19,)))
    # 2D
    p = (1, 1, 4, 5) # nu1, nu2, T, m
    input_size, expected_size = (25, 54), (20, 20)
    func = getattr(dadi.Demographics2D, "split_mig")
    pts_l = [40, 50]

    func_ex = dadi.Numerics.make_extrap_func(func)
    fs = func_ex(p, input_size, pts_l)

    projected_fs = project_fs(fs)
    np.testing.assert_array_equal(projected_fs, fs.project((19, 19)))

