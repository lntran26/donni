""" Tests for train.py """
import os
import pickle
import math
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint, loguniform
from mapie.regression import MapieRegressor
from dadinet.train import prep_data, tune, get_best_specs, train


def test_exists():
    """ Test program exists """

    PRG = '../dadinet/train.py'
    assert os.path.isfile(PRG)


def test_prep_data():
    """ Test prep_data() method """

    for data_file in ['two_epoch_500', 'split_mig_1500']:
        data = pickle.load(open(f'test_data/{data_file}', 'rb'))
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


def run_tune(data_file, param_dist, max_iter=243, eta=3, cv=5):
    """ Template method for testing tune() method """

    data = pickle.load(open(f'{data_file}', 'rb'))
    X, y = prep_data(data)
    all_results = tune(X, y, param_dist, max_iter, eta, cv)
    # check that we have one result per mlpr label
    assert len(all_results) == len(y)
    # check that each band within hyperband for each model
    # is a HalvingRandomSearchCV object (successive halving)
    for each_model in all_results:
        assert all(isinstance(each_band, HalvingRandomSearchCV)
                   for each_band in each_model)
    # check that each model has the correct number of bands
    s_max = int(math.log(max_iter)/math.log(eta))
    assert all(len(results) == s_max+1 for results in all_results)


def test_run_tune1():
    ''' Test tuning with two_epoch '''

    data = 'test_data/two_epoch_500'
    param_dist = {'hidden_layer_sizes': [(64,), (64, 64)],
                  'activation': ['tanh', 'relu'],
                  'solver': ['lbfgs', 'adam'],
                  'alpha': loguniform(1e-4, 1e0),
                  'early_stopping': [False, True]}
    run_tune(data, param_dist, max_iter=27)


def test_run_tune2():
    ''' Test tuning with split_mig'''

    data = 'test_data/split_mig_100_subset'
    param_dist = {'hidden_layer_sizes': [(randint.rvs(50, 100),),
                                         (randint.rvs(50, 100),
                                          randint.rvs(50, 100)),
                                          (randint.rvs(50, 100),
                                          randint.rvs(50, 100),
                                          randint.rvs(50, 100)),
                                          (randint.rvs(50, 100),
                                          randint.rvs(50, 100),
                                          randint.rvs(50, 100),
                                          randint.rvs(50, 100))],
                  'activation': ['tanh', 'relu'],
                  'solver': ['lbfgs', 'adam'],
                  'early_stopping': [False, True]}
    run_tune(data, param_dist, max_iter=25, eta=5)


def test_get_best_specs():
    ''' Test get_best_specs() method '''

    for data_file in ['two_epoch_500', 'split_mig_1500']:
        # this data is tune based on mapie regressor, so 1 mlpr/param
        tune_results = pickle.load(
            open(f'test_data/{data_file}_tune_results_full', 'rb'))
        specs, scores = get_best_specs(tune_results)
        # test that specs is a list of dictionary
        assert isinstance(specs, list)
        assert all(isinstance(spec, dict) for spec in specs)
        # test that scores is a list of float scores
        assert isinstance(scores, list)
        assert all(isinstance(score, float) for score in scores)
        # test that specs and scores have the same length
        assert len(specs) == len(scores)


def run_train(data_file, mlpr_specs, mapie=True):
    """ Template method for testing train() method """

    data = pickle.load(open(f'test_data/{data_file}', 'rb'))
    X, y = prep_data(data)  # mapie=True by default

    trained_mlpr = train(X, y, mlpr_specs, mapie=mapie)

    # test if trained_mlpr and specs are list of the same length
    assert isinstance(trained_mlpr, list)
    assert len(trained_mlpr) == len(mlpr_specs) == len(y)
    # test if output model is a mapie regressor
    assert all(isinstance(mlpr, MapieRegressor) for mlpr in trained_mlpr)
    # # test if output model has the specified specs
    # for mlpr, spec in zip(trained_mlpr, specs):
    #     assert isinstance(mlpr, MapieRegressor)
    #     assert spec.items() <= mlpr.get_params().items()
    # this doesn't work for mapie regressor bc the param dict is different

    # test if output model make a sensible prediction

    # test mapie = False flag to generate regular mlpr with sklearn


def test_run_train1():
    ''' Test train with two_epoch and results from tuning'''

    data = 'two_epoch_500'
    tune_results = pickle.load(
        open(f'test_data/{data}_tune_results_full', 'rb'))
    specs, _ = get_best_specs(tune_results)
    # note: these tune results are created with mapie=True
    run_train(data, specs)


def test_run_train2():
    ''' Test train with split_mig and specified hyperparams'''

    data = 'split_mig_100_subset'
    spec = {'hidden_layer_sizes': (100,),
            'activation': 'relu', 'solver': 'adam'}
    specs = [spec, spec, spec, spec]
    run_train(data, specs)


# if __name__ == "__main__":

#     mlpr_param = {'hidden_layer_sizes': (100,),
#                   'activation': 'relu', 'solver': 'lbfgs',
#                   'alpha': 1, 'max_iter': 5000}
#     data = '../tests/test_data/1d_2epoch_100fs_exclude'
#     ml_model = '../out'

#     # train(mlpr_param, ml_model, data)
#     train(mlpr_param, ml_model, data, mapie=False)
