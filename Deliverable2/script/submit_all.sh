#!/bin/bash

MTX_DIR="./mtx"
BIN_DIR="./bin"

for exe in "$BIN_DIR"/*; do
  exe_name=$(basename "$exe")
  
  for matrix in "$MTX_DIR"/*.mtx; do
    sbatch script/run.sbatch "$exe_name" "$matrix"
  done
done
