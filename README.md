# GPU-Computing-2025-258855

This project is developed as part of the **GPU Computing** course (2024-2025) taught by **Professor Flavio Vella** at the **University of Trento**.  
It focuses on implementing and evaluating a GPU-based **Sparse Matrix-Vector multiplication (SpMV)** using **global memory optimization techniques**.

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

To run all tests on all available `.mtx` files, use:

~~~bash
make test
~~~

This command executes the `script/submit_all.sh` script, which performs the following steps:
- Iterates over every matrix file in the `mtx/` directory.
- For each matrix, it launches all compiled executables (`main`, `opt_main`, `kernel_v1` to `kernel_v4`).
- Submits each test job using `sbatch`, via a shared SLURM script: `script/run.sbatch`.

The `run.sbatch` file includes job scheduling directives and defines how the executable is launched on the HPC system:

~~~bash
#!/bin/bash

#SBATCH --partition=edu-short
#SBATCH --nodelist=edu01
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=output/test-%j.out
#SBATCH --error=output_err/test-%j.err

module load CUDA/12.3.2

EXEC=$1
MTX_FILE=$2

srun ./bin/$EXEC $MTX_FILE
~~~

All scripts, including `submit_all.sh` and `run.sbatch`, are located in the `script/` folder.

This system ensures consistent, parallelized testing and performance data collection.

---

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
├── Makefile
├── README.md
└── report.pdf
```

---

## Notes
A detailed analysis of the project, including implementation choices and performance evaluation, is available in the [project report](./report.pdf).

---