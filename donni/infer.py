'''Module for using trained MLPR to make demographic param predictions'''
import numpy as np
import dadi
import math
from donni.generate_data import pts_l_func

# irods packages
from irods.session import iRODSSession
from irods.models import Collection, DataObject
import irods.exception as exception
from appdirs import AppDirs
import os, shutil, pkg_resources

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


def estimate_theta(pred, func, fs):
    # (p, func, ns, pts_l, folded) = args
    if not fs.folded:
        func = dadi.Numerics.make_anc_state_misid_func(func)
    func_ex = dadi.Numerics.make_extrap_func(func)
    grid_pts = pts_l_func(fs.sample_sizes)
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

def irods_download(username, password, dem_model, sample_sizes, fold=False, tempdir=AppDirs("donni", "Linh Tran", version=pkg_resources.get_distribution("donni").version).user_cache_dir, cleanup=False):
    # Prep naming for model configuration directory
    # If polarization is determined by a flag
    if fold:
        polarization = 'folded'
    else:
        polarization = 'unfolded'

    # Name model configuration directory
    tempdir = tempdir + f"/{dem_model}_{polarization}_ns_{'_'.join([str(ele) for ele in sample_sizes])}"

    # Start irods with the Cyvers Data Store
    session = iRODSSession(host='data.cyverse.org', port=1247, user=username, password=password, zone='iplant')


    # Work with a directory
    try:
        coll = session.collections.get(f"/iplant/home/rgutenkunst/donni/{dem_model}/{polarization}/ss_{'_'.join([str(ele) for ele in sample_sizes])}/tuned_models")
    except exception.CollectionDoesNotExist:
        print("Files for the requested model and configuration do not exist see <blank> for avalible models and configurations")
        # Exit, might break downstream functions
        # Will error if not running as a script
        from sys import exit
        exit()

    # Make model data directory
    try:
        os.makedirs(tempdir)
    except:
        pass

    try:
        # Download files in directory
        for do in coll.data_objects:
            session.data_objects.get(do.path, tempdir, force=True)
            print(f"Downloading: {do.path} to {tempdir}")
    except exception.OVERWRITE_WITHOUT_FORCE_FLAG:
        # If we have users name a folder rather than making it automaticly named based on their requested model and configuration:
        # print("Files for the requested model and configuration have already been downloaded to the given folder, delete the files or rename directory")
        print(f"Files for the requested model and configuration have already been downloaded to the {tempdir} folder, if you want to redownload delete the directory")
    print(f"Finished downloading files to {tempdir} folder")
    return tempdir


def irods_cleanup(dem_model, sample_sizes, unfold=True):

    # Prep naming for model configuration directory
    # If polarization is determined by a flag
    if unfold:
        polarization = 'unfolded'
    else:
        polarization = 'folded'

    # Name model configuration directory
    tempdir = tempdir + f"/{dem_model}_{polarization}_ns_{'_'.join([str(ele) for ele in sample_sizes])}"

    # Remove downloaded files
    # Seperate into a cleanup function for actual donni code?
    try:
        shutil.rmtree(tempdir)
    except FileNotFoundError:
        print("Directory for model configuration not found.")


def infer(models: list, func, input_fs, logs, mapie=True, pis=[95]):
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

    # get input_fs ready for ml prediction
    fs = prep_fs_for_ml(input_fs)

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
        if sum([ele < 0 for ele in pred_list]) > 0:
            raise ValueError("Model inferred a negative parameter value - try a different model.")
        else:
            theta = estimate_theta(pred_list, func, input_fs)

    else:  # sklearn multioutput case: don't know if this works yet
        pred_list = models[0].predict([input_x])
        pi_list = None
        # log transformed prediction results
        pred_list = [10**pred_list[i] if logs[i] else pred_list[i]
                     for i in range(len(logs))]
    return pred_list, theta, pi_list