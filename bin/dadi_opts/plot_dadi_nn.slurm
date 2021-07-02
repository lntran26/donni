#!/bin/bash
#SBATCH --job-name=plot_dadi
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=standard
### REQUIRED. Set the number of cores that will be used for this job.
### SLURM: Each task is assume to require one cpu. 
### Optionally, you may include cpus-per-task if more are required. 
### Requests NxM cpus
#SBATCH --ntasks=1
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=128gb
#SBATCH --time=5:00:00

python plot_dadi_nn.py