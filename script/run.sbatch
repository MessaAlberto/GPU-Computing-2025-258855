#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=output/test-%j.out
#SBATCH --error=output_err/test-%j.err

module load CUDA/12.1.1

# srun ./bin/main ./mtx/nos2.mtx
srun ./bin/main ./mtx/test.mtx
# srun ./bin/tmp ../mtx/nos2.mtx