# GPU-Computing-2025-258855

This repository contains the project developed for the **GPU Computing** course (2024-2025) taught by **Professor Flavio Vella** at the **University of Trento**.  
The project explores **Sparse Matrix-Vector multiplication (SpMV)** on GPU, progressively applying optimizations across two deliverables.

### Project Structure
- **Deliverable 1**  
  Focused on **global memory optimization**:
  - Coalesced memory access.  
  - Baseline comparisons with CPU (sequential + OpenMP).  
  - Evaluation of execution time, FLOPS, and memory throughput.  

- **Deliverable 2**  
  Extended the work with **shared memory** and **advanced CUDA features**:
  - Shared memory tiling and bank conflict minimization.  
  - Warp shuffle intrinsics and loop unrolling.  
  - Comparison with cuSPARSE and Deliverable 1 kernels.  
  - Deeper profiling and architectural insights.  

---

Each deliverable is contained in its own folder, including:
- A dedicated **README** explaining the objectives, structure, compilation, and testing instructions.  
- A **report** describing in detail the implementations, the methodology, and the experimental results.  

To execute the code, please follow the instructions provided in the README of the corresponding deliverable.  
