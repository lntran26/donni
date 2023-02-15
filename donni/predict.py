'''Module for using trained MLPR to make demographic param predictions'''
import numpy as np
import dadi
import math

def get_supported_ss(dims):
    # can update these lists as needed (or pull from cloud)
    if dims < 3:
        return [10, 20, 40, 80, 160]
    else:
        return [10, 20, 40]

def project_fs(input_fs):
    bounds = input_fs.shape
    # take min of the sample sizes as the limit
    bound_limit = min(bounds)
    dims = len(bounds) # 1d, 2d, or 3d
    ss_list = get_supported_ss(dims)
    ss_max = -1
    for ss in ss_list:
        # fs increases dimensions by 1
        if ss + 1 > ss_max and ss + 1 <= bound_limit:
            ss_max = ss
    if ss_max == -1:
        raise ValueError("Sample sizes in input fs are too small")
    # the size is actually -1
    projected_fs = input_fs.project([ss_max for i in range(dims)])
    return projected_fs


def get_grid_pts(ss):
    pts = []
    for i,s in enumerate(ss):
        pt = math.floor(s * (1 + .1*(i+1))) + (2 * (i+1))
        pts.append(pt)
    return pts


def estimate_theta(pred, func, fs):
    # (p, func, ns, pts_l, folded) = args
    if not fs.folded:
        func = dadi.Numerics.make_anc_state_misid_func(func)
    func_ex = dadi.Numerics.make_extrap_func(func)
    grid_pts = get_grid_pts(fs.sample_sizes)
    model_fs = func_ex(pred, fs.sample_sizes, grid_pts)
    return dadi.Inference.optimal_sfs_scaling(model_fs, fs)


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


def predict(models: list, func, input_fs, logs, mapie=True, pis=[95]):
    '''
    Inputs:
        models: list of single mlpr object if sklearn,
            list of multiple mlpr objects if mapie
        input_fs: single Spectrum object from which to generate prediction
        logs: list of bools, indicates which dem params are in log10 values
        if mapie, should be passing in a list of models trained on
            individual params
        if not mapie, should be list of length 1
        pis: list of confidence intervals to calculate
    Outputs:
        pred_list: if mapie, outputs list prediction for each param
        pi_list: if mapie, outputs list of prediction intervals for each
            alpha for each param
    '''
    # project to supported sample sizes
    projected_fs = project_fs(input_fs)

    # get input_fs ready for ml prediction
    fs = prep_fs_for_ml(projected_fs)

    # flatten input_fs and put in a list
    input_x = [np.array(fs).flatten()]

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
        
        theta = estimate_theta(pred_list, func, projected_fs)

    else:  # sklearn multioutput case: don't know if this works yet
        pred_list = models[0].predict([input_x])
        pi_list = None
        # log transformed prediction results
        pred_list = [10**pred_list[i] if logs[i] else pred_list[i]
                     for i in range(len(logs))]
    return pred_list, theta, pi_list