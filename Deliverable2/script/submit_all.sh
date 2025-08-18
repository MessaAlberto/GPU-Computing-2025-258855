#!/bin/bash

MTX_DIR="./mtx"


for matrix in "$MTX_DIR"/*.mtx; do

  for i in {1..4}; do
    sbatch script/run.sbatch kernel_v"$i" "$matrix" 256
  done
done