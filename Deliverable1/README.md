# GPU-Computing-2025-258855 – Deliverable 1

This is the **first part of the project** for the *GPU Computing* course (2024-2025) taught by **Professor Flavio Vella** at the **University of Trento**.  

The focus of this deliverable is on implementing and evaluating a GPU-based **Sparse Matrix-Vector multiplication (SpMV)** using **global memory optimization techniques**.

## Project Structure

```text
├── mtx/                # Downloaded matrix files (MTX format)
├── src/                # Source code (main.c, kernel.cu)
├── lib/                # Support library source files
├── include/            # Header files
├── script/
│   ├── download_mtx.sh
│   ├── submit_all.sh
│   └── run.sbatch
├── py/
│   ├── analyze_blocksize_effect.py # Analyze block size effect
│   ├── comp_results.py             # Compare SpMV on CPU and GPU kernels
│   └── generate_csv.py             # Generate CSV files from output/
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
- CPU and GPU source files from `src/`

The compilation produces the following executables:
- CPU:
  - `main`
  - `opt_main`
- GPU:
  - `kernel_v1`
  - `kernel_v2`
  - `kernel_v3`
  - `kernel_v4`

Each GPU kernel implements a different global memory optimization strategy.

### 3. Run the Tests

There are two types of tests available:

1. **Standard test:**  
   Run with
   ```bash
   make test
   ```
   This executes the `script/submit_all.sh` script, which:
   - Iterates over every matrix file in the `mtx/` directory.
   - For each matrix, launches all compiled executables:
     - `main` and `opt_main` (CPU-based)
     - `kernel_v1` to `kernel_v4` (GPU-based)
   - Submits each test job using `sbatch` and the appropriate SLURM scripts.

   In this mode, all GPU kernels use a fixed `BLOCK_DIM` of 256 threads.

2. **Block size sweep test:**  
   Run with
   ```bash
   make test_block
   ```
   This runs tests on a single matrix defined in `script/submit_blocksize.sh`.  
   It executes all kernels with varying block sizes: 32, 64, 128, 256, ..., up to 1024 threads.

Below is an example of the GPU SLURM submission script `script/run_gpu.sbatch` used by both test types:

```bash
#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:a30.24:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test-gpu
#SBATCH --output=output/test-%j.out
#SBATCH --error=output_err/test-%j.err

module load CUDA/12.3.2

EXEC=$1
MTX_FILE=$2
BLOCK_SIZE=$3

srun ./bin/$EXEC $MTX_FILE $BLOCK_SIZE
```

All submission and SLURM scripts are located in the `script/` folder.

---

## Plotting Results

After running tests, you can generate performance plots using the Python scripts in `py/`.

1. Make sure you have a Python environment ready, either on the cluster or locally.

2. Run the test(s) you want:
   ```bash
   make test # standard kernel tests
   make test_block # block-size variation tests
   ```

3. After each test, generate the CSV from the raw output:
   ```bash
   python3 py/generate_csv.py
   ```
   This creates `results.csv` in the `results/` directory.

4. Generate the plots:
   - For standard kernel results:
   ```bash
   python3 py/comp_results.py
   ```
   This plots execution time, GFLOPS, and bandwidth across all matrices.
   
   - For block-size analysis:
   ```bash
   python3 py/analyze_blocksize_effect.py
   ```
   This plots bandwidth vs block size for the selected matrix.

5. Ensure the `output/` folder contains the correct `.out` files before running these scripts.
   
All generated plots are saved in the `./results/` directory.

---

## Notes
A detailed analysis of the project, including implementation choices and performance evaluation, is available in the [project report](./Deliverable1_report.pdf).

---