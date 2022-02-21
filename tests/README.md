## Running test suites

Changes to the code should be run through the test suites using command `make test` to ensure passing of all tests:
```
$ make test
pytest -xv --flake8 *.py
============================================ test session starts =============================================
platform darwin -- Python 3.9.6, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
cachedir: .pytest_cache
rootdir: ../dadi-ml
plugins: flake8-1.0.7, mypy-0.8.1, timeout-1.4.2, pylint-0.18.0
collected 16 items                                                                                           

test_dadi_dem_models.py::FLAKE8 SKIPPED (file(s) previously passed FLAKE8 checks)                      [  6%]
test_generate_data.py::FLAKE8 PASSED                                                                   [ 12%]
test_generate_data.py::test_exists PASSED                                                              [ 18%]
test_generate_data.py::test_run_two_epoch PASSED                                                       [ 25%]
test_generate_data.py::test_run_growth PASSED                                                          [ 31%]
test_generate_data.py::test_run_split_mig PASSED                                                       [ 37%]
test_generate_data.py::test_run_IM PASSED                                                              [ 43%]
test_integration.py::FLAKE8 PASSED                                                                     [ 50%]
test_integration.py::test_usage PASSED                                                                 [ 56%]
test_integration.py::test_usage_subcommand PASSED                                                      [ 62%]
test_integration.py::test_run_generate_data_sub_1 PASSED                                               [ 68%]
test_integration.py::test_run_generate_data_sub_2 PASSED                                               [ 75%]
test_main.py::FLAKE8 SKIPPED (file(s) previously passed FLAKE8 checks)                                 [ 81%]
test_main.py::test_exists PASSED                                                                       [ 87%]
test_main.py::test_run_generate_data PASSED                                                            [ 93%]
test_train.py::FLAKE8 SKIPPED (file(s) previously passed FLAKE8 checks)                                [100%]

======================================= 13 passed, 3 skipped in 38.88s =======================================
pylint *.py

--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)
```
Alternatively, individual tests can be run with the corresponding commands (see `Makefile`)

