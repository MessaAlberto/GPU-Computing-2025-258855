#!/bin/bash

MTX_DIR="./mtx"


for matrix in "$MTX_DIR"/*.mtx; do

  [ -f "$matrix" ] || continue
  sbatch script/run.sbatch main "$matrix"
  sbatch script/run.sbatch opt_main "$matrix"

  for i in {1..4}; do
    sbatch script/run_gpu.sbatch kernel_v"$i" "$matrix" 256
  done
done