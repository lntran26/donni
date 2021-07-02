#!/bin/bash
#SBATCH --job-name=make_bootstrap
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
### SLURM: Each task is assume to require one cpu. 
### Optionally, you may include cpus-per-task if more are required. 
### Requests NxM cpus
#SBATCH --ntasks=50
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --time=05:00:00

python 2d_splitmig_bootstrap_data.py