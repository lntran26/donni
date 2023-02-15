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



Update most recent `make test` run
```
$ make test
pytest -xv --flake8 *.py
===================================================== test session starts ======================================================
platform darwin -- Python 3.9.6, pytest-6.2.4, py-1.10.0, pluggy-0.13.1 -- /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
cachedir: .pytest_cache
rootdir: /Users/linhtran/Library/CloudStorage/OneDrive-UniversityofArizona/projects/dadi-ml
plugins: flake8-1.0.7, mypy-0.8.1, timeout-1.4.2, pylint-0.18.0
collected 33 items                                                                                                             

test_generate_data.py::FLAKE8 SKIPPED (file(s) previously passed FLAKE8 checks)                                          [  3%]
test_generate_data.py::test_exists PASSED                                                                                [  6%]
test_generate_data.py::test_run_two_epoch PASSED                                                                         [  9%]
test_generate_data.py::test_run_two_epoch_non_norm PASSED                                                                [ 12%]
test_generate_data.py::test_run_two_epoch_folded PASSED                                                                  [ 15%]
test_generate_data.py::test_run_growth PASSED                                                                            [ 18%]
test_generate_data.py::test_run_split_mig PASSED                                                                         [ 21%]
test_generate_data.py::test_run_split_mig_non_norm PASSED                                                                [ 24%]
test_generate_data.py::test_run_split_mig_folded PASSED                                                                  [ 27%]
test_generate_data.py::test_run_IM PASSED                                                                                [ 30%]
test_generate_data.py::test_run_bstr_theta_1 PASSED                                                                      [ 33%]
test_generate_data.py::test_run_two_epoch_bstr PASSED                                                                    [ 36%]
test_generate_data.py::test_run_split_mig_bstr PASSED                                                                    [ 39%]
test_integration.py::FLAKE8 SKIPPED (file(s) previously passed FLAKE8 checks)                                            [ 42%]
test_integration.py::test_usage PASSED                                                                                   [ 45%]
test_integration.py::test_usage_subcommand PASSED                                                                        [ 48%]
test_integration.py::test_run_generate_data_sub_1 PASSED                                                                 [ 51%]
test_integration.py::test_run_generate_data_sub_2 PASSED                                                                 [ 54%]
test_integration.py::test_run_generate_data_sub_folded PASSED                                                            [ 57%]
test_integration.py::test_run_generate_data_bstr PASSED                                                                  [ 60%]
test_integration.py::test_run_train_sub_1 PASSED                                                                         [ 63%]
test_integration.py::test_run_train_sub_2 PASSED                                                                         [ 66%]
test_integration.py::test_run_train_sub_3 PASSED                                                                         [ 69%]
test_predict.py::FLAKE8 SKIPPED (file(s) previously passed FLAKE8 checks)                                                [ 72%]
test_predict.py::test_exists PASSED                                                                                      [ 75%]
test_train.py::FLAKE8 PASSED                                                                                             [ 78%]
test_train.py::test_exists PASSED                                                                                        [ 81%]
test_train.py::test_prep_data PASSED                                                                                     [ 84%]
test_train.py::test_run_tune1 PASSED                                                                                     [ 87%]
test_train.py::test_run_tune2 PASSED                                                                                     [ 90%]
test_train.py::test_get_best_specs PASSED                                                                                [ 93%]
test_train.py::test_run_train1 PASSED                                                                                    [ 96%]
test_train.py::test_run_train2 PASSED                                                                                    [100%]

======================================================= warnings summary =======================================================
tests/test_train.py::test_get_best_specs
tests/test_train.py::test_run_train1
  Trying to unpickle estimator MLPRegressor from version 1.0.1 when using version 1.0.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
  https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations

tests/test_train.py::test_get_best_specs
tests/test_train.py::test_run_train1
  Trying to unpickle estimator HalvingRandomSearchCV from version 1.0.1 when using version 1.0.2. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:
  https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations

-- Docs: https://docs.pytest.org/en/stable/warnings.html
==================================== 30 passed, 3 skipped, 4 warnings in 171.71s (0:02:51) =====================================
pylint *.py
************* Module test_train
test_train.py:5:0: W0611: Unused enable_halving_search_cv imported from sklearn.experimental (unused-import)

------------------------------------------------------------------
Your code has been rated at 9.95/10 (previous run: 9.76/10, +0.19)

make: *** [test] Error 4
```