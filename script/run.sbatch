#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --account=vella.edu
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=output/test-%j.out
#SBATCH --error=output_err/test-%j.err

EXEC=$1
MTX_FILE=$2

srun ./bin/$EXEC $MTX_FILE