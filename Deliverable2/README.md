# GPU-Computing-2025-258855 – Deliverable 2

This is the **second part of the project**, building upon Deliverable 1.  
This part focuses on further improving SpMV performance on the GPU by leveraging **shared memory** and other **advanced CUDA features**. It builds upon the kernel designs from Deliverable 1, adding further optimizations and enhancements.

## Project Structure

```bash
├── include/            # Header files
├── lib/                # Support library source files
├── mtx/                # Downloaded matrix files (MTX format)
├── py/
│   ├── comp_deliverable_results.py   # Compare results from Deliverable 1 and 2
│   ├── comp_kernel_results.py        # Compare results from different kernel implementations
│   └── generate_csv.py               # Generate CSV files from the output data
├── results/            # .csv and .png files
├── script/
│   ├── download_mtx.sh
│   ├── run.sbatch
│   └── submit_all.sh
├── src/                # Source code
│   ├── coalescedBins.cu
│   ├── cuparse.cu
│   ├── hybrid.cu
│   ├── oneThreadPerRow.cu
│   └── oneWarpPerRow.cu
├── Makefile
├── README.md
└── report.pdf
```
---

## How to Run the Project

### 1. Download Test Matrices

First, download selected matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) by running:

```bash
make download
```

This runs the `script/download_mtx.sh` script, which includes preselected matrix links. All matrices will be stored in the `mtx/` folder.

### 2. Compile the Project

Compile everything with:

```bash
make
```

This builds:
- Libraries from `lib/*.c` and headers from `include/*.h`
- CUDA source files from `src/*.cu`

The compilation produces the following executables in the `bin/` directory:
- `coalescedBins`
- `cuparse`
- `hybrid`
- `oneThreadPerRow`
- `oneWarpPerRow`

Each GPU kernel is an optimized implementation of the corresponding GPU kernel provided in Deliverable 1.

### 3. Run the Tests
Launching the command:
```bash
 make test
```
This will execute the `script/submit_all.sh` script, which will test each kernel on every matrix found in `mtx/`. Jobs are submitted to SLURM using the script/run.sbatch script.
```bash
#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=output/test-%j.out
#SBATCH --error=output_err/test-%j.err

module load CUDA/12.3.2

EXEC=$1
MTX_FILE=$2

srun ./bin/$EXEC $MTX_FILE
```

---

## Plotting Results

After running tests, you can generate performance plots using the Python scripts in `py/`:

1. Make sure you have a Python environment ready, either on the cluster or locally.

2. After running the tests and ensuring the `output/` folder contains the `.out` files, proceed with the following steps.

3. Generate the CSV from the raw output:
   ```bash
   python3 py/generate_csv.py
   ```
   This creates `results.csv` in the `results/` directory.

4. Generate the plots:
   - For standard kernel results:
   ```bash
   python3 py/comp_kernel_results.py
   ```
   This plots execution time, GFLOPS, and bandwidth across all matrices.

   - To compare the improvements from Deliverable 1:
   ```bash
   python3 py/comp_deliverable_results.py
   ```
   This plots execution time, GFLOPS, and bandwidth comparing Deliverable 1 and 2.

All generated plots are saved in the `./results/` directory.
---

## Notes
A detailed analysis of the project, including implementation choices and performance evaluation, is available in the [project report](./Deliverable2_report.pdf).


During the development of the project, the plot [`bandwidth_minBlockPerSM_plot.png`](./results/bandwidth_minBlockPerSM_plot.png) was generated. This plot was created by running experiments on various launch-bound implementations (modifying the `minBlockPerSM` parameter). Based on the results, the conditions for the function [`suggest_minBlocksPerSM`](./lib/mtx_utils.c) in `.lib/mtx_utils.c` were determined.

---