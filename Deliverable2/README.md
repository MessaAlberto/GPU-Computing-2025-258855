# GPU-Computing-2025-258855 – Deliverable 2

This is the **second part of the project**, building upon Deliverable 1.  
Here, the focus is on further improving SpMV performance on GPU by using **shared memory** and other **advanced CUDA features**.

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
│   └── compare_spmv_methods.py     # Compare SpMV on CPU and GPU kernels
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
TODO

### 3. Run the Tests
TODO

---

## Plotting Results

After running tests, you can generate performance plots using the Python scripts in `py/`:

- Load a Python environment e.g., on the cluster or transfer to your local machine

- If you ran
  ```bash
  make test
  ```
  run from the root directory:
  ```bash
  python3 py/compare_spmv_methods.py
  ```
  This plots comparisons across all matrices.

- If you ran
  ```bash
  make test_block
  ```
  run from the root directory:
  ```bash
  python3 py/analyze_blocksize_effect.py
  ```
  This plots bandwidth vs block size for the selected matrix.

Both scripts read output data from the `output/` directory.  
**Ensure the `output/` folder contains the correct `.out` files before running these scripts.**

Generated plots are saved to the `./img/` directory.

---

## Notes
<!-- A detailed analysis of the project, including implementation choices and performance evaluation, is available in the [project report](./Deliverable1_report.pdf). -->

---