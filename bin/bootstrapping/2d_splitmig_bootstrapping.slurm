#!/bin/bash
#SBATCH --job-name=bootstrapping
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=standard
#SBATCH --mail-type=ALL
#SBATCH --mail-user=conniesun@email.arizona.edu
### REQUIRED. Set the number of cores that will be used for this job.
### SLURM: Each task is assume to require one cpu. 
### Optionally, you may include cpus-per-task if more are required. 
### Requests NxM cpus
#SBATCH --ntasks=10
### REQUIRED. Set the memory required for this job.
#SBATCH --mem=128gb
#SBATCH --nodes=1
#SBATCH --time=05:00:00

python 2d_splitmig_bootstrapping.py

