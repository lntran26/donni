#!/bin/bash
#SBATCH --job-name=benchmarking_array
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=rgutenk
#SBATCH --partition=standard
### The below command specifies high_priority partition
#SBATCH --qos=user_qos_rgutenk
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=320gb
#SBATCH --time=10:00:00
#SBATCH --array=1-60

date
PROJECT_DIR="/groups/rgutenk/lnt/ml-dadi"
PROGRAM_DIR="$PROJECT_DIR/bin"

python "$PROGRAM_DIR"/benchmarking_parallel.py
date

