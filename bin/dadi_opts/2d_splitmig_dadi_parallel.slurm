#!/bin/bash
#SBATCH --job-name=2d_splitmig_dadi
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
### SLURM: Each task is assume to require one cpu. 
### Optionally, you may include cpus-per-task if more are required. 
### Requests NxM cpus
#SBATCH --ntasks=200
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=256gb
#SBATCH --time=10:00:00

python 2d_splitmig_dadi_parallel.py

