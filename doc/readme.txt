Explanation of directories and files
* note that the path strings in several scripts need to be updated because many scripts/files were moved around

bin/nn/ contains the MLPRegressor scripts
    1d_2epoch_nn.py runs an entire workflow by generating training and testing data, training several NNs based on the theta list, and
    testing the NNs on each set of theta testing data. Prediction plots are saved.
    2d_splitmig_make_*.py files make the specificied test/train data sets for the split mig model. 
    2d_splitmig_nn.py is the main script for testing and plotting results of the MLPRegressors 
        the other 2d_splitmig_nn_*.py scripts are variations on the main method (e.g., logTm is training/testing on the logs of T and m)
    gridsearchcv.py is used to search hyperparameters for improving NN accuracy
    util.py has the most updated methods used for the NN
    all *.slurm files match with their associated file names, except the make_* files truncated the model name to be shorter 
    
bin/rf/ contains the RandomForestRegressor scripts
    currently, all old versions of the full workflow process for the two-epoch and split mig models
        2d_splitmig_log.py uses the logs of nu1 and nu2 for training, testing, and plotting

bin/bootstrapping/ contains the scripts to produce and analyze bootstrap data
    2d_splitmig_bootstrap_data.py generates all of the data for bootstrapping (original params, pseudo-sampled fs, bootstrap fs)
    2d_splitmig_bootstrapping.py contains various methods for obtaining ML predictions on bootstrap data, creating confidence intervals,
    and visualizing results

bin/dadi_opts/ contains the scripts used to run many dadi optimizations and compare results to the ML methods
    2d_splitmig_dadi_parallel.py does a single optimization off of the test set that is used for the ML algorithms
    2d_splitmig_dadi_optimize.py saves dadi optimizations based off of ten random starting points 
    2d_splitmig_dadi_optimize_full.py attempts to continue optimizing while not converged for a max of five opts
        don't use this script; it's not good
    2d_splitmig_dadi_converged.py checks for convergence based on a certain epsilon value for the top 3 "best" optimizations (log 
    likelihood), and will run an additional 9 optimizations if not converged (perturbs 3 times based on top 3 best results). Convergence
    is checked again and the top three are saved.
    plot_dadi.py is an old script for plotting dadi true vs. pred results that did not check for convergence
    plot_dadi_converged.py plots dadi true vs. pred results that are converged based on the epsilon specified in the above scripts; it 
    also contains code to plot the ratios of m and nus 
    plot_dadi_nn.py plots dadi ored vs. nn pred results 
    test_opt.py is mostly useless (checks that dadi.Inference.opt is installed correctly and is a reference for dadi specifications)
    
data/ contains "global" data
    all of the files in this directory are pickle files that are loaded and used by scripts in bin/
    for example, it contains NNs trained on various thetas, as well as test and train sets
    note: the *_results* files in this directory are placed in this directory because they are loaded and used by other scripts; files in 
    the results/ directory (described below) are "endpoint" results that are not used by other scripts
    
results/ contains various results from running different scripts (.txt, .out, .png, and .pdf formats)
    mostly contains plots from training/testing; the names should be informative