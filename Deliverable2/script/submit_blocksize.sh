#!/bin/bash

MTX_FILE="./mtx/ecology1.mtx"

for BLOCK_SIZE in 32 64 128 256 512 1024; do
  sbatch script/run_gpu.sbatch cuparse "$MTX_FILE" "$BLOCK_SIZE"
  
  for i in {1..4}; do
    sbatch script/run_gpu.sbatch kernel_v"$i" "$MTX_FILE" "$BLOCK_SIZE"
  done
done
