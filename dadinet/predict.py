'''Module for using trained MLPR to make demographic param predictions'''
import numpy as np


def prep_fs_for_ml(input_fs):
    '''normalize and set masked entries to zeros
    input_fs: single Spectrum object from which to generate prediction'''
    # make sure the input_fs is normalized
    if round(input_fs.sum(), 3) != float(1):
        input_fs = input_fs/input_fs.sum()
    # assign zeros to masked entries of fs
    input_fs.flat[0] = 0
    input_fs.flat[-1] = 0

    return input_fs


def predict(models: list, input_fs, logs, mapie=True, pis=[95]):
    '''
    models: list of single mlpr object if sklearn,
        list of multiple mlpr objects if mapie
    input_fs: single Spectrum object from which to generate prediction

    if mapie, should be passing in a list of models trained on
        individual params
    if not mapie, should be list of length 1
    '''
    # TODO: check fs shape and project down to exp input size if needed

    # get input_fs ready for ml prediction
    input_fs = prep_fs_for_ml(input_fs)

    # flatten input_fs and put in a list
    input_x = [np.array(input_fs).flatten()]

    # convert intervals to decimals
    alpha = [(100 - pi) / 100 for pi in pis]

    # get prediction using trained ml models
    if mapie:
        pred_list = []
        pi_list = []
        for i, model in enumerate(models):
            pred, pis = model.predict(input_x, alpha=alpha)
            pred = pred[0]  # one sample
            pis = pis[0]    # one sample
            if logs[i]:
                pred = 10 ** pred
                pis = 10 ** pis
            pred_list.append(pred)
            pi_list.append(pis.T)

    else:  # sklearn multioutput case: don't know if this works yet
        pred_list = models[0].predict([input_x])
        pi_list = None
        # log transformed prediction results
        pred_list = [10**pred_list[i] if logs[i] else pred_list[i]
                     for i in range(len(logs))]
    return pred_list, pi_list
