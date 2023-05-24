#!/usr/bin/env python
###SBATCH --job-name=hpc_pipeline
#SBATCH --job-name=tunetrain_ooa
#SBATCH --output=hpc_output/%x-%A_%a.out
#SBATCH --account=rgutenk
#SBATCH --qos=user_qos_rgutenk
#SBATCH --mail-type=ALL
###SBATCH --mail-user=tjstruck@arizona.edu
###SBATCH --partition=windfall
#SBATCH --partition=high_priority
#SBATCH --nodes=1
#SBATCH --ntasks=90
###SBATCH --cpus-per-task=1
###SBATCH --mem-per-cpu=5gb
###SBATCH --constraint=hi_mem
###SBATCH --mem-per-cpu=32gb
#SBATCH --time=48:00:00
###SBATCH --array=1-136
###SBATCH --array=1-2
import sys,os
# Python-specific stuff
print('Script running\n')
if 'SLURM_SUBMIT_DIR' in os.environ:
#       # Set module search path so it will search in qsub directory
      sys.path.insert(0, os.environ['SLURM_SUBMIT_DIR'])
#       # Set current working directory to qsub directory
      # os.chdir(os.environ['SLURM_SUBMIT_DIR'])
# Which process am I?
process_ii = int(os.environ.get('SLURM_ARRAY_TASK_ID',1))-1

# Define version number as a variable
version = "v0.0.1"

# Make list of all dadi and portik models
import dadi
from inspect import getmembers, isfunction
from donni.custom_models import out_of_africa
import time
import glob
portik_2d_models = []
portik_3d_models = []
try:
    import portik_models_2d
    import portik_models_3d
    portik_2d_models = [m[0] for m in getmembers(portik_models_2d, isfunction)]
    portik_3d_models = [m[0] for m in getmembers(portik_models_3d, isfunction)]
except:
    print("Portik models not in directory")

duplicated_models = ["snm", "bottlegrowth"]

oned_models = [m[0] for m in getmembers(dadi.Demographics1D, isfunction)]
twod_models = [m[0] for m in getmembers(dadi.Demographics2D, isfunction)]

for m in duplicated_models:
    oned_models.remove(m)
for m in duplicated_models:
    twod_models.remove(m)
oned_models
twod_models.remove("IM_mscore")
twod_models.remove("IM_pre_mscore")
twod_models.remove("split_mig_mscore")

model_l = oned_models + twod_models + portik_2d_models + portik_3d_models + ["out_of_africa"]

# Set up stats for pipeline
pipeline_l = []
ns_l = [10, 20, 40, 80, 160]

# Loop through models
for model_e in model_l:
    for ns_e in ns_l:
        # loop through sample sizes
        # Convert sample sizes to length required for the model
        if model_e == 'out_of_africa':
            ns_e = [ns_e, ns_e, ns_e]
        elif model_e in oned_models:
            ns_e = [ns_e]
        elif model_e in portik_3d_models:
            ns_e = [ns_e, ns_e, ns_e]
        else:
            ns_e = [ns_e, ns_e]
        for fold_flag_e in ['--folded', '']:
            # Skip 3D models over ns 20
            if ns_e not in [[40, 40, 40], [80 ,80 ,80], [160, 160, 160]]:
                pipeline_l.append([model_e, ns_e, fold_flag_e])

model, ns, fold_flag = pipeline_l[process_ii]

print(f"Need {len(pipeline_l)} jobs to run desired setups")
training_samples = 5000
testing_samples = 1000
testing_theta = 1000
training_theta = 1


# Makr directories
# model/fold_dir/ss_dir/version/
#                   ..> /plots/
#                   ..> /tuned_models/

# Define if a directory name is folded or unfolded
if fold_flag == '--folded':
    fold_dir = 'folded'
else:
    fold_dir = 'unfolded'

ss_dir = 'ss_' + '_'.join([str(ele) for ele in ns])
ss_flag = ' '.join([str(ele) for ele in ns])

# Define directory paths for donni commands
base_dir = "/".join([model, fold_dir, ss_dir, version])
mlpr_dir = f"{base_dir}/tuned_models/"
plot_dir = f"{base_dir}/plots/"

# Dictionary path/name for pickled objects used multiple times
training_fid = f"{base_dir}/data/train_{training_samples}_theta_{training_theta}"
testing_fid = f"{base_dir}/data/test_{testing_samples}_theta_{testing_theta}"
param_dict_tune = f"{base_dir}/param_dict_tune"

# Make directories
os.makedirs(f"{base_dir}/data/", exist_ok=True)
os.makedirs(f"{mlpr_dir}", exist_ok=True)
os.makedirs(f"{plot_dir}", exist_ok=True)

# Make testing data
start = time.time()
os.system(f"donni generate_data --seed 100 --theta {testing_theta} --model {model} --n_samples {testing_samples} \
    --sample_sizes {ss_flag} --outfile {testing_fid} {fold_flag}")
print(f"Finish: {time.time() - start}")

# Make training data
start = time.time()
os.system(f"donni generate_data --seed 1 --theta {training_theta} --model {model} --n_samples {training_samples} \
    --sample_sizes {ss_flag} --generate_tune_hyperparam --hyperparam_outfile {param_dict_tune} --outfile {training_fid} {fold_flag}")
print(f"Finish: {time.time() - start}")

# Tune and Train
print("Tuning and training")
# Run if tunning
os.system(f"donni train --data_file {training_fid} --mlpr_dir {mlpr_dir} --tune --max_iter 300 \
    --hyperparam {param_dict_tune} --cv 5 --eta 3 --hyperparam_list {mlpr_dir}/tuned_hyperparam_dict_list")

# Plot
print("Plotting")
os.system(f"donni plot --model {model} --theta {testing_theta} --test_dict {testing_fid} --mlpr_dir {mlpr_dir} \
    --train_dict {training_fid} --results_dir {plot_dir} --plot_prefix theta_1000 --coverage")


# Upload to Cyvers
print("Uploading to Cyverse")
from irods.session import iRODSSession
from irods.models import Collection, DataObject
import irods.exception as exception
from appdirs import AppDirs
import glob

# Fill in username and password
username = ''
password = ''

session = iRODSSession(host='data.cyverse.org', port=1247, user=username, password=password, zone='iplant')

session.collections.create(f"/iplant/home/shared/donni/{model}/{fold_dir}/{ss_dir}/{version}/tuned_models/")
session.collections.create(f"/iplant/home/shared/donni/{model}/{fold_dir}/{ss_dir}/{version}/plots/")

fids = glob.glob(f"{mlpr_dir}/param_*") + glob.glob(f"{plot_dir}/*")

for fid in fids:
    session.data_objects.put(fid, '/iplant/home/shared/donni/'+fid)








